import logging

# Add TRACE logging level (below DEBUG)
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")

def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kwargs)

logging.Logger.trace = trace

# Configure logging with TRACE enabled
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=TRACE_LEVEL  # Enable TRACE and all higher levels
)
