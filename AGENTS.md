# ECLlib Codex Instructions

## Environment
- Prefer the repository virtual environment `.venv_ECLlib` when running Python, `pip`, or `pytest` commands in this repository.
- Use `.venv_ECLlib/bin/python`, `.venv_ECLlib/bin/pip`, and `.venv_ECLlib/bin/pytest` when the environment exists.

## Python Class And Function Formatting
When adding or modifying Python classes, functions, or methods in this repository:

- Keep separator lines exactly 100 characters wide including indentation.
- Do not add duplicate separator lines if they already exist.
- Add a concise, informative docstring to any new function or method that does not already have one.

### Classes In `src/`
- Insert a `#====...` separator line immediately before each `class` definition.
- Insert the same `#====...` separator line immediately after the class header line.
- Add a right-aligned trailing comment with the class name on the class header line.

Required class format:

```python
#===================================================================================================
class ClassName:                                                                         # ClassName
#===================================================================================================
```

### Methods In `src/`
- Insert a `#----...` separator line immediately before each method definition.
- Insert the same `#----...` separator line immediately after the method signature line.
- Add a right-aligned trailing comment with the owning class name on the method header line.
- If the method header already needs an important inline comment such as `# pragma: no cover`, preserve it and append `| ClassName`.
- Do not apply these method tags to functions defined inside another function, even if they appear inside a local class.

Required method format:

```python
    #-----------------------------------------------------------------------------------------------
    def method_name(self) -> None:                                                       # ClassName
    #-----------------------------------------------------------------------------------------------
        """Explain what the method does."""
```

### Top-Level And Nested Functions
- For functions outside classes, use plain `#----...` separator lines before and after the definition/signature line.
- Do not add a trailing class-name comment to top-level or nested non-method functions.
- Do not tag functions defined within functions.
- In Python terminology these are usually called local functions or nested functions. They are closures only when they capture values from the enclosing scope.
- This plain function format applies throughout the repository, including `tests/` and `scripts/`.

Required function format:

```python
#---------------------------------------------------------------------------------------------------
def example(arg1, arg2):
#---------------------------------------------------------------------------------------------------
    """Explain what the function does."""
```

Local nested function format:

```python
def helper(value):
    """Explain what the local helper does."""
```

