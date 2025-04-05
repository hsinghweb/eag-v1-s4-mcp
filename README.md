# MCP (Mathematical Computation and Presentation) System

A powerful system that combines mathematical computations with automated PowerPoint presentation generation. The system consists of three main components: a server providing mathematical tools, a client for direct interaction, and an AI agent for natural language processing.

## Components

### MCP Server (mcp-server.py)
- Provides various mathematical functions including:
  - Basic arithmetic operations (add, subtract, multiply, divide)
  - Advanced math functions (power, square root, cube root, factorial, logarithm)
  - Trigonometric functions (sin, cos, tan)
  - Special functions (ASCII conversion, exponential sum, Fibonacci sequence)
- PowerPoint automation capabilities:
  - Create and manage presentations
  - Draw shapes (rectangles)
  - Add and format text
  - Automated presentation handling

### MCP Client (mcp-client.py)
- Establishes connection with the MCP server
- Demonstrates usage of server tools
- Example implementation shows:
  - Converting text to ASCII values
  - Calculating exponential sums
  - Creating PowerPoint presentations with results

### AI Agent (ai-agent.py)
- Provides natural language interface to the system
- Interprets user queries and executes appropriate tools
- Manages multi-step operations through iterations
- Handles both mathematical calculations and presentation creation

## Setup and Usage

1. Ensure Python is installed on your system
2. Run the server:
   ```
   python mcp-server.py dev
   ```
3. Run either the client or AI agent:
   ```
   python mcp-client.py
   # or
   python ai-agent.py
   ```

## Example Operations

- Mathematical calculations
- ASCII value conversion
- Exponential computations
- Automated PowerPoint creation
- Natural language query processing (via AI agent)

## Requirements

- Python 3.x
- PowerPoint installation (for presentation features)
- Required Python packages (mcp, pywinauto, python-pptx)
