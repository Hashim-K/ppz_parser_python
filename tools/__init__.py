import pkgutil
import inspect

# Import the base class that all tools inherit from.
from .base_tool import BaseTool

# This dictionary will hold all the tool classes, ready to be used.
# The key is the user-friendly name provided in the decorator, and the value is the class itself.
TOOL_REGISTRY = {}


def register_tool(name: str):
    """
    A decorator that registers a tool class in the TOOL_REGISTRY with a specific name.
    """

    def decorator(cls):
        # Use the provided name as the key in the registry.
        TOOL_REGISTRY[name] = cls
        return cls

    return decorator


# --- Auto-discovery of tools ---
# This code block automatically imports all modules in the current package (the 'tools' directory)
# so that their @register_tool decorators run and populate the TOOL_REGISTRY.

# Get the current package name.
package = __import__(__name__, fromlist=[""])

# Iterate over all modules in this package.
for _, module_name, _ in pkgutil.iter_modules(package.__path__):
    # Import the module. This executes the code in the module file, including the decorators.
    __import__(f"{__name__}.{module_name}", fromlist=[""])
