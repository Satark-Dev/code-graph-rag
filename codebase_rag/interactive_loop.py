from __future__ import annotations

import asyncio
import os
import shlex
import shutil
from collections.abc import Coroutine
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import print_formatted_text
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.text import Text

from . import constants as cs
from . import logs as ls
from .bootstrap import _create_model_from_string
from .config import ModelConfig, settings
from .context import app_context
from .session_logging import get_session_context, init_session_log, log_session_event
from .types_defs import (
    CHAT_LOOP_UI,
    OPTIMIZATION_LOOP_UI,
    ORANGE_STYLE,
    AgentLoopUI,
    CancelledResult,
    ConfirmationToolNames,
)
from .ui import style
from .prompts import OPTIMIZATION_PROMPT, OPTIMIZATION_PROMPT_WITH_REFERENCE

if TYPE_CHECKING:
    from prompt_toolkit.key_binding import KeyPressEvent
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models import Model


async def run_optimization_loop(
    rag_agent: Agent[None, str],
    message_history: list[ModelMessage],
    project_root: Path,
    language: str,
    tool_names: ConfirmationToolNames,
    reference_document: str | None = None,
) -> None:
    app_context.console.print(cs.UI_OPTIMIZATION_START.format(language=language))
    document_info = (
        cs.UI_REFERENCE_DOC_INFO.format(reference_document=reference_document)
        if reference_document
        else ""
    )
    app_context.console.print(
        Panel(
            cs.UI_OPTIMIZATION_PANEL.format(document_info=document_info),
            border_style=cs.Color.YELLOW,
        )
    )

    initial_question = (
        OPTIMIZATION_PROMPT_WITH_REFERENCE.format(
            language=language, reference_document=reference_document
        )
        if reference_document
        else OPTIMIZATION_PROMPT.format(language=language)
    )

    await _run_interactive_loop(
        rag_agent,
        message_history,
        project_root,
        OPTIMIZATION_LOOP_UI,
        style(cs.PROMPT_YOUR_RESPONSE, cs.Color.CYAN),
        tool_names,
        initial_question,
    )


async def run_with_cancellation[T](
    coro: Coroutine[None, None, T], timeout: float | None = None
) -> T | CancelledResult:
    task = asyncio.create_task(coro)

    try:
        return await asyncio.wait_for(task, timeout=timeout) if timeout else await task
    except TimeoutError:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        app_context.console.print(
            f"\n{style(cs.MSG_TIMEOUT_FORMAT.format(timeout=timeout), cs.Color.YELLOW)}"
        )
        return CancelledResult(cancelled=True)
    except (asyncio.CancelledError, KeyboardInterrupt):
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        app_context.console.print(
            f"\n{style(cs.MSG_THINKING_CANCELLED, cs.Color.YELLOW)}"
        )
        return CancelledResult(cancelled=True)


async def _run_agent_response_loop(
    rag_agent: Agent[None, str],
    message_history: list[ModelMessage],
    question_with_context: str,
    config: AgentLoopUI,
    tool_names: ConfirmationToolNames,
    model_override: Model | None = None,
) -> None:
    from pydantic_ai import DeferredToolRequests, DeferredToolResults, ToolDenied

    deferred_results: DeferredToolResults | None = None

    while True:
        with app_context.console.status(config.status_message):
            response = await run_with_cancellation(
                rag_agent.run(
                    question_with_context,
                    message_history=message_history,
                    deferred_tool_results=deferred_results,
                    model=model_override,
                ),
            )

        if isinstance(response, CancelledResult):
            log_session_event(config.cancelled_log)
            app_context.session.cancelled = True
            break

        if isinstance(response.output, DeferredToolRequests):
            deferred_results = _process_tool_approvals(
                response.output,
                config.approval_prompt,
                config.denial_default,
                tool_names,
            )
            message_history.extend(response.new_messages())
            continue

        output_text = response.output
        if not isinstance(output_text, str):
            continue
        markdown_response = Markdown(output_text)
        app_context.console.print(
            Panel(
                markdown_response,
                title=config.panel_title,
                border_style=cs.Color.GREEN,
            )
        )

        log_session_event(f"{cs.SESSION_PREFIX_ASSISTANT}{output_text}")
        message_history.extend(response.new_messages())
        break


def _process_tool_approvals(
    requests,
    approval_prompt: str,
    denial_default: str,
    tool_names: ConfirmationToolNames,
):
    from pydantic_ai import DeferredToolResults, ToolDenied
    from .types_defs import RawToolArgs
    from .ui import dim

    deferred_results = DeferredToolResults()

    for call in requests.approvals:
        tool_args = _to_tool_args(
            call.tool_name, RawToolArgs(**call.args_as_dict()), tool_names
        )
        app_context.console.print(
            f"\n{cs.UI_TOOL_APPROVAL.format(tool_name=call.tool_name)}"
        )
        _display_tool_call_diff(call.tool_name, tool_args, tool_names)

        if app_context.session.confirm_edits:
            if Confirm.ask(style(approval_prompt, cs.Color.CYAN)):
                deferred_results.approvals[call.tool_call_id] = True
            else:
                feedback = Prompt.ask(
                    cs.UI_FEEDBACK_PROMPT,
                    default="",
                )
                denial_msg = feedback.strip() or denial_default
                deferred_results.approvals[call.tool_call_id] = ToolDenied(denial_msg)
        else:
            deferred_results.approvals[call.tool_call_id] = True

    return deferred_results


def _to_tool_args(tool_name: str, raw_args, tool_names: ConfirmationToolNames):
    from .types_defs import (
        CreateFileArgs,
        ReplaceCodeArgs,
        ShellCommandArgs,
    )

    match tool_name:
        case tool_names.replace_code:
            return ReplaceCodeArgs(
                file_path=raw_args.file_path,
                target_code=raw_args.target_code,
                replacement_code=raw_args.replacement_code,
            )
        case tool_names.create_file:
            return CreateFileArgs(
                file_path=raw_args.file_path,
                content=raw_args.content,
            )
        case tool_names.shell_command:
            return ShellCommandArgs(command=raw_args.command)
        case _:
            return ShellCommandArgs()


def _display_tool_call_diff(
    tool_name: str,
    tool_args,
    tool_names: ConfirmationToolNames,
    file_path: str | None = None,
) -> None:
    from .types_defs import ToolArgs
    from .ui import dim

    match tool_name:
        case tool_names.replace_code:
            target = str(tool_args.get(cs.ARG_TARGET_CODE, ""))
            replacement = str(tool_args.get(cs.ARG_REPLACEMENT_CODE, ""))
            path = str(
                tool_args.get(cs.ARG_FILE_PATH, file_path or cs.DIFF_FALLBACK_PATH)
            )
            _print_unified_diff(target, replacement, path)

        case tool_names.create_file:
            path = str(tool_args.get(cs.ARG_FILE_PATH, ""))
            content = str(tool_args.get(cs.ARG_CONTENT, ""))
            _print_new_file_content(path, content)

        case tool_names.shell_command:
            command = tool_args.get(cs.ARG_COMMAND, "")
            app_context.console.print(f"\n{cs.UI_SHELL_COMMAND_HEADER}")
            app_context.console.print(
                style(f"$ {command}", cs.Color.YELLOW, cs.StyleModifier.NONE)
            )

        case _:
            app_context.console.print(
                cs.UI_TOOL_ARGS_FORMAT.format(
                    args=json.dumps(tool_args, indent=cs.JSON_INDENT)
                )
            )


def _print_unified_diff(target: str, replacement: str, path: str) -> None:
    import difflib
    from .ui import dim

    separator = dim(cs.HORIZONTAL_SEPARATOR)
    app_context.console.print(f"\n{cs.UI_DIFF_FILE_HEADER.format(path=path)}")
    app_context.console.print(separator)

    diff = difflib.unified_diff(
        target.splitlines(keepends=True),
        replacement.splitlines(keepends=True),
        fromfile=cs.DIFF_LABEL_BEFORE,
        tofile=cs.DIFF_LABEL_AFTER,
        lineterm="",
    )

    for line in diff:
        line = line.rstrip("\n")
        match line[:1]:
            case cs.DiffMarker.ADD | cs.DiffMarker.DEL if line.startswith(
                cs.DiffMarker.HEADER_ADD
            ) or line.startswith(cs.DiffMarker.HEADER_DEL):
                app_context.console.print(dim(line))
            case cs.DiffMarker.HUNK:
                app_context.console.print(
                    style(line, cs.Color.CYAN, cs.StyleModifier.NONE)
                )
            case cs.DiffMarker.ADD:
                app_context.console.print(
                    style(line, cs.Color.GREEN, cs.StyleModifier.NONE)
                )
            case cs.DiffMarker.DEL:
                app_context.console.print(
                    style(line, cs.Color.RED, cs.StyleModifier.NONE)
                )
            case _:
                app_context.console.print(line)

    app_context.console.print(separator)


def _print_new_file_content(path: str, content: str) -> None:
    from .ui import dim

    separator = dim(cs.HORIZONTAL_SEPARATOR)
    app_context.console.print(f"\n{cs.UI_NEW_FILE_HEADER.format(path=path)}")
    app_context.console.print(separator)

    for line in content.splitlines():
        app_context.console.print(
            style(f"{cs.DiffMarker.ADD} {line}", cs.Color.GREEN, cs.StyleModifier.NONE)
        )

    app_context.console.print(separator)


def get_multiline_input(prompt_text: str = cs.PROMPT_ASK_QUESTION) -> str:
    bindings = KeyBindings()

    @bindings.add(cs.KeyBinding.CTRL_J)
    def submit(event: KeyPressEvent) -> None:
        event.app.exit(result=event.app.current_buffer.text)

    @bindings.add(cs.KeyBinding.ENTER)
    def new_line(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add(cs.KeyBinding.CTRL_C)
    def keyboard_interrupt(event: KeyPressEvent) -> None:
        event.app.exit(exception=KeyboardInterrupt)

    clean_prompt = Text.from_markup(prompt_text).plain

    print_formatted_text(
        HTML(
            cs.UI_INPUT_PROMPT_HTML.format(
                prompt=clean_prompt, hint=cs.MULTILINE_INPUT_HINT
            )
        )
    )

    result = prompt(
        "",
        multiline=True,
        key_bindings=bindings,
        wrap_lines=True,
        style=ORANGE_STYLE,
    )
    if result is None:
        raise EOFError
    stripped: str = result.strip()
    return stripped


def _handle_model_command(
    command: str,
    current_model: Model | None,
    current_model_string: str | None,
    current_config: ModelConfig | None,
) -> tuple[Model | None, str | None, ModelConfig | None]:
    parts = command.strip().split(maxsplit=1)
    arg = parts[1].strip() if len(parts) > 1 else None

    if not arg:
        if current_model_string:
            display_model = current_model_string
        else:
            config = settings.active_orchestrator_config
            display_model = f"{config.provider}{cs.CHAR_COLON}{config.model_id}"
        app_context.console.print(cs.UI_MODEL_CURRENT.format(model=display_model))
        return current_model, current_model_string, current_config

    if arg.lower() == cs.HELP_ARG:
        app_context.console.print(cs.UI_MODEL_USAGE)
        return current_model, current_model_string, current_config

    try:
        new_model, canonical_model_string, new_config = _create_model_from_string(
            arg, current_config
        )
        logger.info(ls.MODEL_SWITCHED.format(model=canonical_model_string))
        app_context.console.print(
            cs.UI_MODEL_SWITCHED.format(model=canonical_model_string)
        )
        return new_model, canonical_model_string, new_config
    except (ValueError, AssertionError) as e:
        logger.error(ls.MODEL_SWITCH_FAILED.format(error=e))
        app_context.console.print(cs.UI_MODEL_SWITCH_ERROR.format(error=e))
        return current_model, current_model_string, current_config


async def _run_interactive_loop(
    rag_agent: Agent[None, str],
    message_history: list[ModelMessage],
    project_root: Path,
    config: AgentLoopUI,
    input_prompt: str,
    tool_names: ConfirmationToolNames,
    initial_question: str | None = None,
) -> None:
    init_session_log(project_root)
    question = initial_question or ""
    model_override: Model | None = None
    model_override_string: str | None = None
    model_override_config: ModelConfig | None = None

    while True:
        try:
            if not initial_question or question != initial_question:
                question = await asyncio.to_thread(get_multiline_input, input_prompt)

            stripped_question = question.strip()
            stripped_lower = stripped_question.lower()

            if stripped_lower in cs.EXIT_COMMANDS:
                break

            if not stripped_question:
                initial_question = None
                continue

            command_parts = stripped_lower.split(maxsplit=1)
            if command_parts[0] == cs.MODEL_COMMAND_PREFIX:
                model_override, model_override_string, model_override_config = (
                    _handle_model_command(
                        stripped_question,
                        model_override,
                        model_override_string,
                        model_override_config,
                    )
                )
                initial_question = None
                continue
            if command_parts[0] == cs.HELP_COMMAND:
                app_context.console.print(cs.UI_HELP_COMMANDS)
                initial_question = None
                continue

            log_session_event(f"{cs.SESSION_PREFIX_USER}{question}")

            if app_context.session.cancelled:
                question_with_context = question + get_session_context()
                app_context.session.reset_cancelled()
            else:
                question_with_context = question

            question_with_context = _handle_chat_images(
                question_with_context, project_root
            )

            await _run_agent_response_loop(
                rag_agent,
                message_history,
                question_with_context,
                config,
                tool_names,
                model_override,
            )

            initial_question = None

        except KeyboardInterrupt:
            break
        except Exception as e:  # noqa: BLE001
            logger.exception(ls.UNEXPECTED.format(error=e))
            app_context.console.print(cs.UI_ERR_UNEXPECTED.format(error=e))


def _find_image_paths(question: str) -> list[Path]:
    try:
        if os.name == "nt":
            tokens = shlex.split(question, posix=False)
        else:
            tokens = shlex.split(question)
    except ValueError:
        tokens = question.split()

    image_paths: list[Path] = []
    for token in tokens:
        token = token.strip("'\"")
        if token.lower().endswith(cs.IMAGE_EXTENSIONS):
            p = Path(token)
            if p.is_absolute() or token.startswith("/") or token.startswith("\\"):
                image_paths.append(p)
    return image_paths


def _get_path_variants(path_str: str) -> tuple[str, ...]:
    return (
        path_str.replace(" ", r"\ "),
        f"'{path_str}'",
        f'"{path_str}"',
        path_str,
    )


def _replace_path_in_question(question: str, old_path: str, new_path: str) -> str:
    for variant in _get_path_variants(old_path):
        if variant in question:
            return question.replace(variant, new_path)
    logger.warning(ls.PATH_NOT_IN_QUESTION.format(path=old_path))
    return question


def _handle_chat_images(question: str, project_root: Path) -> str:
    image_files = _find_image_paths(question)
    if not image_files:
        return question

    tmp_dir = project_root / cs.TMP_DIR
    tmp_dir.mkdir(exist_ok=True)
    updated_question = question

    for original_path in image_files:
        if not original_path.exists() or not original_path.is_file():
            logger.warning(ls.IMAGE_NOT_FOUND.format(path=original_path))
            continue

        try:
            new_path = tmp_dir / f"{uuid.uuid4()}-{original_path.name}"
            shutil.copy(original_path, new_path)
            new_relative = str(new_path.relative_to(project_root))
            updated_question = _replace_path_in_question(
                updated_question, str(original_path), new_relative
            )
            logger.info(ls.IMAGE_COPIED.format(path=new_relative))
        except Exception as e:  # noqa: BLE001
            logger.error(ls.IMAGE_COPY_FAILED.format(error=e))

    return updated_question


async def run_chat_loop(
    rag_agent: Agent[None, str],
    message_history: list[ModelMessage],
    project_root: Path,
    tool_names: ConfirmationToolNames,
) -> None:
    await _run_interactive_loop(
        rag_agent,
        message_history,
        project_root,
        CHAT_LOOP_UI,
        style(cs.PROMPT_ASK_QUESTION, cs.Color.CYAN),
        tool_names,
    )

