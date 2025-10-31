# Graph Tools Tests

**Real integration tests** for LangGraph tools - no mocks, actual Modal execution and S3 operations.

## Test Files

- `test_sandbox_tools.py` - Real integration tests for Modal-based sandbox execution and dataset tools

## What These Tests Actually Do

These are **real integration tests** that:
- ✅ Execute Python code in actual Modal sandboxes
- ✅ Call deployed Modal functions (.remote())
- ✅ Upload and download datasets to/from real S3
- ✅ Test persistent state across code executions
- ✅ Verify executor lifecycle management

**No mocks** except for the ToolRuntime framework object (which is just metadata plumbing).

## Requirements

### For Sandbox Tools Tests

You need **Modal deployed** and **S3** configured:

```bash
# Required environment variables
MODAL_TOKEN_ID=<your-modal-token-id>
MODAL_TOKEN_SECRET=<your-modal-token-secret>
S3_BUCKET=<your-s3-bucket>
AWS_ACCESS_KEY_ID=<your-aws-key>
AWS_SECRET_ACCESS_KEY=<your-aws-secret>
AWS_REGION=<your-region>
```

**Important**: Deploy the Modal functions first:
```bash
cd /home/matteo/LG-Urban
modal deploy backend/modal_runtime/functions.py
```

### No Postgres Required

Unlike other integration tests, the sandbox tools tests **do not require Postgres**. They only test:
- Modal sandbox execution (real)
- S3 operations (real)
- Executor lifecycle management

The `thread_id` context is set programmatically without needing database sessions.

## Running the Tests

### Run all graph tests:
```bash
cd /home/matteo/LG-Urban
pytest new_tests/graph/ -v
```

### Run only sandbox tools tests:
```bash
pytest new_tests/graph/test_sandbox_tools.py -v
```

### Run a specific test:
```bash
pytest new_tests/graph/test_sandbox_tools.py::TestExecuteCodeTool::test_execute_simple_code -v
```

### Run with output visible:
```bash
pytest new_tests/graph/test_sandbox_tools.py -v -s
```
