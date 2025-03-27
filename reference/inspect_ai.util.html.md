# inspect_ai.util


## Store

### Store

The `Store` is used to record state and state changes.

The `TaskState` for each sample has a `Store` which can be used when
solvers and/or tools need to coordinate changes to shared state. The
`Store` can be accessed directly from the `TaskState` via `state.store`
or can be accessed using the `store()` global function.

Note that changes to the store that occur are automatically recorded to
transcript as a `StoreEvent`. In order to be serialised to the
transcript, values and objects must be JSON serialisable (you can make
objects with several fields serialisable using the `@dataclass`
decorator or by inheriting from Pydantic `BaseModel`)

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_store.py#L20)

``` python
class Store
```

#### Methods

get  
Get a value from the store.

Provide a `default` to automatically initialise a named store value with
the default when it does not yet exist.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_store.py#L46)

``` python
def get(self, key: str, default: VT | None = None) -> VT | Any
```

`key` str  
Name of value to get

`default` VT \| None  
Default value (defaults to `None`)

set  
Set a value into the store.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_store.py#L64)

``` python
def set(self, key: str, value: Any) -> None
```

`key` str  
Name of value to set

`value` Any  
Value to set

delete  
Remove a value from the store.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_store.py#L73)

``` python
def delete(self, key: str) -> None
```

`key` str  
Name of value to remove

keys  
View of keys within the store.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_store.py#L81)

``` python
def keys(self) -> KeysView[str]
```

values  
View of values within the store.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_store.py#L85)

``` python
def values(self) -> ValuesView[Any]
```

items  
View of items within the store.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_store.py#L89)

``` python
def items(self) -> ItemsView[str, Any]
```

### store

Get the currently active `Store`.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_store.py#L103)

``` python
def store() -> Store
```

### store_as

Get a Pydantic model interface to the store.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_store_model.py#L101)

``` python
def store_as(model_cls: Type[SMT]) -> SMT
```

`model_cls` Type\[SMT\]  
Pydantic model type (must derive from StoreModel)

### StoreModel

Store backed Pydandic BaseModel.

The model is initialised from a Store, so that Store should either
already satisfy the validation constraints of the model OR you should
provide Field(default=) annotations for all of your model fields (the
latter approach is recommended).

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_store_model.py#L8)

``` python
class StoreModel(BaseModel)
```

## Concurrency

### concurrency

Concurrency context manager.

A concurrency context can be used to limit the number of coroutines
executing a block of code (e.g calling an API). For example, here we
limit concurrent calls to an api (‘api-name’) to 10:

``` python
async with concurrency("api-name", 10):
    # call the api
```

Note that concurrency for model API access is handled internally via the
`max_connections` generation config option. Concurrency for launching
subprocesses is handled via the `subprocess` function.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_concurrency.py#L11)

``` python
@contextlib.asynccontextmanager
async def concurrency(
    name: str,
    concurrency: int,
    key: str | None = None,
) -> AsyncIterator[None]
```

`name` str  
Name for concurrency context. This serves as the display name for the
context, and also the unique context key (if the `key` parameter is
omitted)

`concurrency` int  
Maximum number of coroutines that can enter the context.

`key` str \| None  
Unique context key for this context. Optional. Used if the unique key
isn’t human readable – e.g. includes api tokens or account ids so that
the more readable `name` can be presented to users e.g in console UI\>

### subprocess

Execute and wait for a subprocess.

Convenience method for solvers, scorers, and tools to launch
subprocesses. Automatically enforces a limit on concurrent subprocesses
(defaulting to os.cpu_count() but controllable via the
`max_subprocesses` eval config option).

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_subprocess.py#L71)

``` python
async def subprocess(
    args: str | list[str],
    text: bool = True,
    input: str | bytes | memoryview | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] = {},
    capture_output: bool = True,
    output_limit: int | None = None,
    timeout: int | None = None,
) -> Union[ExecResult[str], ExecResult[bytes]]
```

`args` str \| list\[str\]  
Command and arguments to execute.

`text` bool  
Return stdout and stderr as text (defaults to True)

`input` str \| bytes \| memoryview \| None  
Optional stdin for subprocess.

`cwd` str \| Path \| None  
Switch to directory for execution.

`env` dict\[str, str\]  
Additional environment variables.

`capture_output` bool  
Capture stderr and stdout into ExecResult (if False, then output is
redirected to parent stderr/stdout)

`output_limit` int \| None  
Stop reading output if it exceeds the specified limit (in bytes).

`timeout` int \| None  
Timeout. If the timeout expires then a `TimeoutError` will be raised.

### ExecResult

Execution result from call to `subprocess()`.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_subprocess.py#L27)

``` python
@dataclass
class ExecResult(Generic[T])
```

#### Attributes

`success` bool  
Did the process exit with success.

`returncode` int  
Return code from process exit.

`stdout` T  
Contents of stdout.

`stderr` T  
Contents of stderr.

## Display

### display_counter

Display a counter in the UI.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_display.py#L65)

``` python
def display_counter(caption: str, value: str) -> None
```

`caption` str  
The counter’s caption e.g. “HTTP rate limits”.

`value` str  
The counter’s value e.g. “42”.

### display_type

Get the current console display type.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_display.py#L47)

``` python
def display_type() -> DisplayType
```

### DisplayType

Console display type.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_display.py#L11)

``` python
DisplayType = Literal["full", "conversation", "rich", "plain", "none"]
```

### input_screen

Input screen for receiving user input.

Context manager that clears the task display and provides a screen for
receiving console input.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_console.py#L13)

``` python
@contextmanager
def input_screen(
    header: str | None = None,
    transient: bool | None = None,
    width: int | None = None,
) -> Iterator[Console]
```

`header` str \| None  
Header line to print above console content (defaults to printing no
header)

`transient` bool \| None  
Return to task progress display after the user completes input (defaults
to `True` for normal sessions and `False` when trace mode is enabled).

`width` int \| None  
Input screen width in characters (defaults to full width)

## Subtasks

### subtask

Decorator for subtasks.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_subtask.py#L66)

``` python
def subtask(
    name: str | Subtask,
    store: Store | None = None,
    type: str | None = None,
    input: dict[str, Any] | None = None,
) -> Callable[..., Subtask] | Subtask
```

`name` str \| [Subtask](inspect_ai.util.qmd#subtask)  
Name for subtask (defaults to function name)

`store` [Store](inspect_ai.util.qmd#store) \| None  
Store to use for subtask

`type` str \| None  
Type to use for subtask

`input` dict\[str, Any\] \| None  
Input to log for subtask

### Subtask

Subtask with distinct `Store` and `Transcript`.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_subtask.py#L31)

``` python
class Subtask(Protocol):
    async def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any
```

`*args` Any  
Arguments for the subtask.

`**kwargs` Any  
Keyword arguments for the subtask.

## Utilities

### resource

Read and resolve a resource to a string.

Resources are often used for templates, configuration, etc. They are
sometimes hard-coded strings, and sometimes paths to external resources
(e.g. in the local filesystem or remote stores e.g. s3:// or
<https://>).

The `resource()` function will resolve its argument to a resource
string. If a protocol-prefixed file name (e.g. s3://) or the path to a
local file that exists is passed then it will be read and its contents
returned. Otherwise, it will return the passed `str` directly This
function is mostly intended as a helper for other functions that take
either a string or a resource path as an argument, and want to easily
resolve them to the underlying content.

If you want to ensure that only local or remote files are consumed,
specify `type="file"`. For example:
`resource("templates/prompt.txt", type="file")`

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_resource.py#L9)

``` python
def resource(
    resource: str,
    type: Literal["auto", "file"] = "auto",
    fs_options: dict[str, Any] = {},
) -> str
```

`resource` str  
Path to local or remote (e.g. s3://) resource, or for `type="auto"` (the
default), a string containing the literal resource value.

`type` Literal\['auto', 'file'\]  
For “auto” (the default), interpret the resource as a literal string if
its not a valid path. For “file”, always interpret it as a file path.

`fs_options` dict\[str, Any\]  
Optional. Additional arguments to pass through to the `fsspec`
filesystem provider (e.g. `S3FileSystem`). Use `{"anon": True }` if you
are accessing a public S3 bucket with no credentials.

### throttle

Throttle a function to ensure it is called no more than every n seconds.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_throttle.py#L6)

``` python
def throttle(seconds: float) -> Callable[..., Any]
```

`seconds` float  
Throttle time.

### trace_action

Trace a long running or poentially unreliable action.

Trace actions for which you want to collect data on the resolution
(e.g. succeeded, cancelled, failed, timed out, etc.) and duration of.

Traces are written to the `TRACE` log level (which is just below `HTTP`
and `INFO`). List and read trace logs with `inspect trace list` and
related commands (see `inspect trace --help` for details).

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/_util/trace.py#L32)

``` python
@contextmanager
def trace_action(
    logger: Logger, action: str, message: str, *args: Any, **kwargs: Any
) -> Generator[None, None, None]
```

`logger` Logger  
Logger to use for tracing (e.g. from `getLogger(__name__)`)

`action` str  
Name of action to trace (e.g. ‘Model’, ‘Subprocess’, etc.)

`message` str  
Message describing action (can be a format string w/ args or kwargs)

`*args` Any  
Positional arguments for `message` format string.

`**kwargs` Any  
Named args for `message` format string.

### trace_message

Log a message using the TRACE log level.

The `TRACE` log level is just below `HTTP` and `INFO`). List and read
trace logs with `inspect trace list` and related commands (see
`inspect trace --help` for details).

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/_util/trace.py#L133)

``` python
def trace_message(
    logger: Logger, category: str, message: str, *args: Any, **kwargs: Any
) -> None
```

`logger` Logger  
Logger to use for tracing (e.g. from `getLogger(__name__)`)

`category` str  
Category of trace message.

`message` str  
Trace message (can be a format string w/ args or kwargs)

`*args` Any  
Positional arguments for `message` format string.

`**kwargs` Any  
Named args for `message` format string.

## Sandbox

### sandbox

Get the SandboxEnvironment for the current sample.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_sandbox/context.py#L22)

``` python
def sandbox(name: str | None = None) -> SandboxEnvironment
```

`name` str \| None  
Optional sandbox environmnent name.

### sandbox_with

Get the SandboxEnvironment for the current sample that has the specified
file.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_sandbox/context.py#L52)

``` python
async def sandbox_with(file: str, on_path: bool = False) -> SandboxEnvironment | None
```

`file` str  
Path to file to check for if on_path is False. If on_path is True, file
should be a filename that exists on the system path.

`on_path` bool  
If True, file is a filename to be verified using “which”. If False, file
is a path to be checked within the sandbox environments.

### SandboxEnvironment

Environment for executing arbitrary code from tools.

Sandbox environments provide both an execution environment as well as a
per-sample filesystem context to copy samples files into and resolve
relative paths to.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_sandbox/environment.py#L80)

``` python
class SandboxEnvironment(abc.ABC)
```

#### Methods

exec  
Execute a command within a sandbox environment.

The current working directory for execution will be the per-sample
filesystem context.

Each output stream (stdout and stderr) is limited to 10 MiB. If
exceeded, an `OutputLimitExceededError` will be raised.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_sandbox/environment.py#L87)

``` python
@abc.abstractmethod
async def exec(
    self,
    cmd: list[str],
    input: str | bytes | None = None,
    cwd: str | None = None,
    env: dict[str, str] = {},
    user: str | None = None,
    timeout: int | None = None,
    timeout_retry: bool = True,
) -> ExecResult[str]
```

`cmd` list\[str\]  
Command or command and arguments to execute.

`input` str \| bytes \| None  
Standard input (optional).

`cwd` str \| None  
Current working dir (optional). If relative, will be relative to the
per-sample filesystem context.

`env` dict\[str, str\]  
Environment variables for execution.

`user` str \| None  
Optional username or UID to run the command as.

`timeout` int \| None  
Optional execution timeout (seconds).

`timeout_retry` bool  
Retry the command in the case that it times out. Commands will be
retried up to twice, with a timeout of no greater than 60 seconds for
the first retry and 30 for the second.

write_file  
Write a file into the sandbox environment.

If the parent directories of the file path do not exist they should be
automatically created.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_sandbox/environment.py#L133)

``` python
@abc.abstractmethod
async def write_file(self, file: str, contents: str | bytes) -> None
```

`file` str  
Path to file (relative file paths will resolve to the per-sample working
directory).

`contents` str \| bytes  
Text or binary file contents.

read_file  
Read a file from the sandbox environment.

File size is limited to 100 MiB.

When reading text files, implementations should preserve newline
constructs (e.g. crlf should be preserved not converted to lf). This is
equivalent to specifying `newline=""` in a call to the Python `open()`
function.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_sandbox/environment.py#L159)

``` python
@abc.abstractmethod
async def read_file(self, file: str, text: bool = True) -> Union[str | bytes]
```

`file` str  
Path to file (relative file paths will resolve to the per-sample working
directory).

`text` bool  
Read as a utf-8 encoded text file.

connection  
Information required to connect to sandbox environment.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_sandbox/environment.py#L190)

``` python
async def connection(self) -> SandboxConnection
```

as_type  
Verify and return a reference to a subclass of SandboxEnvironment.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_sandbox/environment.py#L202)

``` python
def as_type(self, sandbox_cls: Type[ST]) -> ST
```

`sandbox_cls` Type\[ST\]  
Class of sandbox (subclass of SandboxEnvironment)

default_concurrency  
Default max_sandboxes for this provider (`None` means no maximum)

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_sandbox/environment.py#L221)

``` python
@classmethod
def default_concurrency(cls) -> int | None
```

task_init  
Called at task startup initialize resources.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_sandbox/environment.py#L226)

``` python
@classmethod
async def task_init(
    cls, task_name: str, config: SandboxEnvironmentConfigType | None
) -> None
```

`task_name` str  
Name of task using the sandbox environment.

`config` SandboxEnvironmentConfigType \| None  
Implementation defined configuration (optional).

sample_init  
Initialize sandbox environments for a sample.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_sandbox/environment.py#L238)

``` python
@classmethod
async def sample_init(
    cls,
    task_name: str,
    config: SandboxEnvironmentConfigType | None,
    metadata: dict[str, str],
) -> dict[str, "SandboxEnvironment"]
```

`task_name` str  
Name of task using the sandbox environment.

`config` SandboxEnvironmentConfigType \| None  
Implementation defined configuration (optional).

`metadata` dict\[str, str\]  
Sample `metadata` field

sample_cleanup  
Cleanup sandbox environments.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_sandbox/environment.py#L259)

``` python
@classmethod
@abc.abstractmethod
async def sample_cleanup(
    cls,
    task_name: str,
    config: SandboxEnvironmentConfigType | None,
    environments: dict[str, "SandboxEnvironment"],
    interrupted: bool,
) -> None
```

`task_name` str  
Name of task using the sandbox environment.

`config` SandboxEnvironmentConfigType \| None  
Implementation defined configuration (optional).

`environments` dict\[str, 'SandboxEnvironment'\]  
Sandbox environments created for this sample.

`interrupted` bool  
Was the task interrupted by an error or cancellation

task_cleanup  
Called at task exit as a last chance to cleanup resources.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_sandbox/environment.py#L278)

``` python
@classmethod
async def task_cleanup(
    cls, task_name: str, config: SandboxEnvironmentConfigType | None, cleanup: bool
) -> None
```

`task_name` str  
Name of task using the sandbox environment.

`config` SandboxEnvironmentConfigType \| None  
Implementation defined configuration (optional).

`cleanup` bool  
Whether to actually cleanup environment resources (False if
`--no-sandbox-cleanup` was specified)

cli_cleanup  
Handle a cleanup invoked from the CLI (e.g. inspect sandbox cleanup).

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_sandbox/environment.py#L292)

``` python
@classmethod
async def cli_cleanup(cls, id: str | None) -> None
```

`id` str \| None  
Optional ID to limit scope of cleanup.

config_files  
Standard config files for this provider (used for automatic discovery)

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_sandbox/environment.py#L301)

``` python
@classmethod
def config_files(cls) -> list[str]
```

config_deserialize  
Deserialize a sandbox-specific configuration model from a dict.

Override this method if you support a custom configuration model.

A basic implementation would be:
`return MySandboxEnvironmentConfig(**config)`

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_sandbox/environment.py#L306)

``` python
@classmethod
def config_deserialize(cls, config: dict[str, Any]) -> BaseModel
```

`config` dict\[str, Any\]  
Configuration dictionary produced by serializing the configuration
model.

### SandboxConnection

Information required to connect to sandbox.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_sandbox/environment.py#L61)

``` python
class SandboxConnection(BaseModel)
```

#### Attributes

`type` str  
Sandbox type name (e.g. ‘docker’, ‘local’, etc.)

`command` str  
Shell command to connect to sandbox.

`vscode_command` list\[Any\] \| None  
Optional vscode command (+args) to connect to sandbox.

`ports` list\[PortMapping\] \| None  
Optional list of port mappings into container

`container` str \| None  
Optional container name (does not apply to all sandboxes).

### sandboxenv

Decorator for registering sandbox environments.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_sandbox/registry.py#L16)

``` python
def sandboxenv(name: str) -> Callable[..., Type[T]]
```

`name` str  
Name of SandboxEnvironment type

## JSON

### JSONType

Valid types within JSON schema.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_json.py#L23)

``` python
JSONType = Literal["string", "integer", "number", "boolean", "array", "object", "null"]
```

### JSONSchema

JSON Schema for type.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_json.py#L27)

``` python
class JSONSchema(BaseModel)
```

#### Attributes

`type` [JSONType](inspect_ai.util.qmd#jsontype) \| None  
JSON type of tool parameter.

`description` str \| None  
Parameter description.

`default` Any  
Default value for parameter.

`enum` list\[Any\] \| None  
Valid values for enum parameters.

`items` Optional\[[JSONSchema](inspect_ai.util.qmd#jsonschema)\]  
Valid type for array parameters.

`properties` dict\[str, [JSONSchema](inspect_ai.util.qmd#jsonschema)\] \| None  
Valid fields for object parametrs.

`additionalProperties` Optional\[[JSONSchema](inspect_ai.util.qmd#jsonschema)\] \| bool \| None  
Are additional properties allowed?

`anyOf` list\[[JSONSchema](inspect_ai.util.qmd#jsonschema)\] \| None  
Valid types for union parameters.

`required` list\[str\] \| None  
Required fields for object parameters.

### json_schema

Provide a JSON Schema for the specified type.

Schemas can be automatically inferred for a wide variety of Python class
types including Pydantic BaseModel, dataclasses, and typed dicts.

[Source](https://github.com/UKGovernmentBEIS/inspect_ai/blob/4d11396dcc8dcd01315501009537c77b4c0c5a93/src/inspect_ai/util/_json.py#L58)

``` python
def json_schema(t: Type[Any]) -> JSONSchema
```

`t` Type\[Any\]  
Python type
