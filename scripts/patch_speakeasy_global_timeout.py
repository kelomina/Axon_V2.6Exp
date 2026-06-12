#!/usr/bin/env python3
"""Patch Speakeasy-X winemu.py to implement global wall-clock timeout.

This script modifies E:\Project\python\Speakeasy-X\speakeasy\windows\winemu.py
to replace the per-run timeout with a global wall-clock timeout budget shared
across all queued runs. This prevents samples with many entry points or
exports from running indefinitely and ensures a report is always generated.
"""

from pathlib import Path

WINEMU_PATH = Path(r"E:\Project\python\Speakeasy-X\speakeasy\windows\winemu.py")

# ── Patch 1: Add `import time` after `import shlex` ──────────────────────────

OLD_IMPORTS = """import logging
import ntpath
import os
import shlex
import traceback"""

NEW_IMPORTS = """import logging
import ntpath
import os
import shlex
import time
import traceback"""

# ── Patch 2: Replace start() method with global timeout version ──────────────

OLD_START = '''    def start(self, addr=None, size=None):
        """
        Begin emulation executing each run in the specified run queue
        """
        try:
            run = self.run_queue.pop(0)
        except IndexError:
            return

        self.run_complete = False
        self.set_hooks()
        self._set_emu_hooks()

        # Initialize run context/register state before exposing the target to GDB,
        # so the first stop reports a meaningful PC/SP/etc.
        self._prepare_run_context(run)

        if self.gdb_port is not None:
            from udbserver import udbserver

            logger.info(
                "GDB server listening on port %d, waiting for connection (initial PC: 0x%x)...",
                self.gdb_port,
                self.curr_run.start_addr,  # type: ignore[union-attr]
            )
            udbserver(self.emu_eng.emu, port=self.gdb_port, start_addr=0)  # type: ignore[union-attr]

        timeout = 0 if self.gdb_port is not None else self.config.timeout

        if self.profiler:
            self.profiler.set_start_time()

        while True:
            try:
                self.curr_mod = self.get_module_from_addr(self.curr_run.start_addr)  # type: ignore[union-attr]
                self.emu_eng.start(self.curr_run.start_addr, timeout=timeout, count=self.config.max_instructions)  # type: ignore[union-attr]
                if self.profiler and timeout > 0:
                    if self.profiler.get_run_time() > timeout:
                        logger.error("* Timeout of %d sec(s) reached.", timeout)
            except KeyboardInterrupt:
                logger.error("* User exited.")
                return
            except Exception as e:
                if self.exit_event and self.exit_event.is_set():
                    return
                stack_trace = traceback.format_exc()

                try:
                    mnem, op, instr = self.get_disasm(self.get_pc(), DISASM_SIZE)
                except Exception as dis_err:
                    logger.error(str(dis_err))

                error = self.get_error_info(str(e), self.get_pc(), traceback=stack_trace)
                self.curr_run.error = error  # type: ignore[union-attr]

                run = self.on_run_complete()
                if not run:
                    break
                if self.profiler and timeout > 0 and self.profiler.get_run_time() > timeout:
                    logger.error("* Timeout of %d sec(s) reached.", timeout)
                    break
                continue
            break

        self.on_emu_complete()'''

NEW_START = '''    def start(self, addr=None, size=None):
        """
        Begin emulation executing each run in the specified run queue.

        Implements a global wall-clock timeout budget: the configured timeout
        value is treated as the total time limit for ALL queued runs combined,
        not a per-run limit. This prevents samples with many entry points or
        exports from running indefinitely and ensures a report is always
        generated when the function returns.
        """
        try:
            run = self.run_queue.pop(0)
        except IndexError:
            return

        self.run_complete = False
        self.set_hooks()
        self._set_emu_hooks()

        # Initialize run context/register state before exposing the target to GDB,
        # so the first stop reports a meaningful PC/SP/etc.
        self._prepare_run_context(run)

        if self.gdb_port is not None:
            from udbserver import udbserver

            logger.info(
                "GDB server listening on port %d, waiting for connection (initial PC: 0x%x)...",
                self.gdb_port,
                self.curr_run.start_addr,  # type: ignore[union-attr]
            )
            udbserver(self.emu_eng.emu, port=self.gdb_port, start_addr=0)  # type: ignore[union-attr]

        # Global wall-clock deadline shared across all queued runs.
        # When GDB is attached (gdb_port set), timeout is disabled (0) so the
        # debugger can pause at its leisure.
        configured_timeout = 0 if self.gdb_port is not None else self.config.timeout
        global_deadline = time.monotonic() + configured_timeout if configured_timeout > 0 else 0.0
        _global_timed_out = False

        if self.profiler:
            self.profiler.set_start_time()

        while True:
            # Check global budget before starting each run.
            if global_deadline > 0:
                remaining = global_deadline - time.monotonic()
                if remaining <= 0:
                    logger.error(
                        "* Global timeout of %d sec(s) reached (runs remaining in queue: %d).",
                        configured_timeout,
                        len(self.run_queue),
                    )
                    _global_timed_out = True
                    break
                run_timeout = max(0.1, remaining)
            else:
                run_timeout = configured_timeout

            try:
                self.curr_mod = self.get_module_from_addr(self.curr_run.start_addr)  # type: ignore[union-attr]
                self.emu_eng.start(self.curr_run.start_addr, timeout=run_timeout, count=self.config.max_instructions)  # type: ignore[union-attr]
                if self.profiler and run_timeout > 0:
                    if self.profiler.get_run_time() > run_timeout:
                        logger.error("* Timeout of %d sec(s) reached.", run_timeout)
            except KeyboardInterrupt:
                logger.error("* User exited.")
                return
            except Exception as e:
                if self.exit_event and self.exit_event.is_set():
                    return
                stack_trace = traceback.format_exc()

                try:
                    mnem, op, instr = self.get_disasm(self.get_pc(), DISASM_SIZE)
                except Exception as dis_err:
                    logger.error(str(dis_err))

                error = self.get_error_info(str(e), self.get_pc(), traceback=stack_trace)
                self.curr_run.error = error  # type: ignore[union-attr]

                run = self.on_run_complete()
                if not run:
                    break
                # Also check global budget after an error-triggered run transition.
                if global_deadline > 0 and time.monotonic() >= global_deadline:
                    logger.error(
                        "* Global timeout of %d sec(s) reached after error transition (runs remaining in queue: %d).",
                        configured_timeout,
                        len(self.run_queue),
                    )
                    _global_timed_out = True
                    break
                continue
            break

        # Tag timed-out runs still in the queue so the report reflects partial
        # coverage. This preserves the queue entries so entry_point_count in the
        # JSON report includes both executed and skipped runs.
        if _global_timed_out:
            for pending_run in self.run_queue:
                if pending_run.error is None:
                    pending_run.error = type(
                        "Run", (), {"error_str": lambda self: f"Global timeout ({configured_timeout}s) - run not executed"}
                    )()

        # Always call on_emu_complete to finalize profiler data and generate
        # the report. Previously this was always called; now it is guaranteed
        # even after a global timeout, ensuring the report is written before
        # the parent process kills this worker.
        self.on_emu_complete()'''


def apply_patches() -> None:
    content = WINEMU_PATH.read_text(encoding="utf-8")

    # Verify the file contains what we expect
    if OLD_IMPORTS not in content:
        raise RuntimeError(
            f"Could not find expected import block in {WINEMU_PATH}. "
            "File may have already been patched or has been modified."
        )

    if OLD_START not in content:
        raise RuntimeError(
            f"Could not find expected start() method in {WINEMU_PATH}. "
            "File may have already been patched or has been modified."
        )

    # Apply patches
    content = content.replace(OLD_IMPORTS, NEW_IMPORTS, 1)
    content = content.replace(OLD_START, NEW_START, 1)

    # Verify patches were applied
    if "import time" not in content:
        raise RuntimeError("import time patch failed.")
    if "global_deadline" not in content:
        raise RuntimeError("global_deadline patch failed.")
    if "time.monotonic()" not in content:
        raise RuntimeError("time.monotonic() patch failed.")

    # Write back
    WINEMU_PATH.write_text(content, encoding="utf-8")

    # Verify by re-reading
    verify = WINEMU_PATH.read_text(encoding="utf-8")
    assert "global_deadline = time.monotonic() + configured_timeout" in verify
    assert "_global_timed_out = False" in verify
    assert "Always call on_emu_complete" in verify

    print(f"[OK] Patches applied successfully to {WINEMU_PATH}")
    print(f"     - Added `import time`")
    print(f"     - Replaced start() with global wall-clock timeout implementation")
    print(f"     - Added timeout tag for pending runs in queue")
    print(f"     - Guaranteed on_emu_complete() call after timeout")


if __name__ == "__main__":
    apply_patches()
