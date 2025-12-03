import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPServerConfig(BaseSettings):
    """Configuration for a single MCP server."""

    name: str = Field(description="Unique name for the MCP server")
    transport: str = Field(
        default="stdio", description="Transport type: stdio, http, sse"
    )
    command: Optional[str] = Field(
        default=None, description="Command to run for stdio transport"
    )
    args: List[str] = Field(
        default_factory=list, description="Arguments for the command"
    )
    url: Optional[str] = Field(default=None, description="URL for http/sse transport")
    env: dict = Field(
        default_factory=dict, description="Environment variables for the server"
    )
    enabled: bool = Field(default=True, description="Whether this server is enabled")

    model_config = SettingsConfigDict(extra="ignore")


class Settings(BaseSettings):
    """Application settings managed by Pydantic."""

    # Cursor/OpenAI Configuration
    # Note: When running in Cursor IDE, these are automatically detected from Cursor's active model
    USE_CURSOR_MODEL: bool = Field(
        default=True,
        description="Use Cursor's active model instead of external API key"
    )
    OPENAI_API_KEY: str = Field(
        default="",
        description="Optional: Only needed if USE_CURSOR_MODEL=False. Leave empty to use Cursor's built-in model."
    )
    OPENAI_BASE_URL: str = Field(
        default="",
        description="Optional: Custom API base URL. Auto-detected from Cursor if USE_CURSOR_MODEL=True."
    )
    MODEL_NAME: str = Field(
        default="",
        description="Optional: Model name. Auto-detected from Cursor's active model if USE_CURSOR_MODEL=True."
    )

    # Agent Configuration
    AGENT_NAME: str = "CursorAgent"
    DEBUG_MODE: bool = False

    # Memory Configuration
    MEMORY_FILE: str = "agent_memory.json"

    # MCP Configuration
    MCP_ENABLED: bool = Field(default=False, description="Enable MCP integration")
    MCP_SERVERS_CONFIG: str = Field(
        default="mcp_servers.json", description="Path to MCP servers configuration file"
    )
    MCP_CONNECTION_TIMEOUT: int = Field(
        default=30, description="Timeout in seconds for MCP server connections"
    )
    MCP_TOOL_PREFIX: str = Field(
        default="mcp_", description="Prefix for MCP tool names to avoid conflicts"
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


# Global settings instance
settings = Settings()
