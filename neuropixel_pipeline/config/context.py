### Example
class ManagedResource:
    def __enter__(self):
        print("Enter: Acquiring resource.")
        return self  # This object is bound to the target of the `with` statement

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exit: Releasing resource.")
        # Handle exception, if any, and return False to propagate or True to suppress

    def operate(self):
        print("Operating on the resource.")

# Usage
with ManagedResource() as resource:
    resource.operate()


### Experimenting
class GenericFilepathContext: # This isn't transparent Dependency Injection (DI)
    def __enter__(self, replacement):
        print("Enter: Acquiring resource.")
        return self  # This object is bound to the target of the `with` statement

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exit: Releasing resource.")
        # Handle exception, if any, and return False to propagate or True to suppress

    def operate(self):
        print("Operating on the resource.")

# Usage
with GenericFilepathContext() as resource:
    resource.operate()


### The config object (returned from pipeline_config()) should really be a Context Manager itself
### There needs to be a way to inject the config per entry (as in per populate calls going through the pipeline).
### It shouldn't persist between entries and therefore between contexts.