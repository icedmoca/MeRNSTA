#!/usr/bin/env python3
"""
🌉 MeRNSTA API Main - Compatibility wrapper for backward compatibility
This module provides backward compatibility for imports and testing.
"""

# Import the main system bridge
from .system_bridge import SystemBridgeAPI

# Create app instance for compatibility with tests and other modules
def create_app():
    """Create FastAPI app instance."""
    bridge = SystemBridgeAPI()
    return bridge.app

# Export the app for backward compatibility
app = create_app()

# Export other important components
__all__ = ["app", "SystemBridgeAPI", "create_app"]