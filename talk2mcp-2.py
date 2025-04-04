import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import asyncio
from google import genai
from concurrent.futures import TimeoutError
from functools import partial
import logging
import sys

# Load environment variables from .env file
load_dotenv()

# Configure event loop policy for Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_agent.log')
    ]
)
logger = logging.getLogger(__name__)

# Access your API key and initialize Gemini client correctly
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Constants and state variables
MAX_ITERATIONS = 10
MAX_RETRIES = 3
TIMEOUT_SECONDS = 10

# State variables


class State:
    def __init__(self):
        self.last_response = None
        self.iteration = 0
        self.iteration_response = []

    def reset(self):
        self.last_response = None
        self.iteration = 0
        self.iteration_response = []


state = State()


async def generate_with_timeout(client, prompt, timeout=TIMEOUT_SECONDS):
    """Generate content with a timeout"""
    logger.info("Starting LLM generation...")
    try:
        loop = asyncio.get_running_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                    generation_config={
                        "temperature": 0.7,
                        "top_p": 0.8,
                        "top_k": 40
                    }
                )
            ),
            timeout=timeout
        )
        logger.info("LLM generation completed successfully")
        return response
    except TimeoutError as e:
        logger.error(f"LLM generation timed out after {timeout} seconds")
        raise
    except Exception as e:
        logger.error(f"Error in LLM generation: {str(e)}")
        raise


async def handle_function_call(session, tools, response_text):
    """Handle function calls from LLM response"""
    try:
        _, function_info = response_text.split(":", 1)
        parts = [p.strip() for p in function_info.split("|")]
        func_name, params = parts[0], parts[1:]

        logger.debug(
            f"Processing function call: {func_name} with params: {params}")

        tool = next((t for t in tools if t.name == func_name), None)
        if not tool:
            raise ValueError(f"Unknown tool: {func_name}")

        arguments = {}
        schema_properties = tool.inputSchema.get('properties', {})

        for param_name, param_info in schema_properties.items():
            if not params:
                raise ValueError(f"Not enough parameters for {func_name}")

            value = params.pop(0)
            param_type = param_info.get('type', 'string')

            arguments[param_name] = await convert_param_value(value, param_type)

        result = await session.call_tool(func_name, arguments=arguments)
        return result

    except Exception as e:
        logger.error(f"Error in function call: {e}")
        raise


async def convert_param_value(value, param_type):
    """Convert parameter value to the correct type with improved error handling"""
    try:
        if not isinstance(value, str):
            raise ValueError(f"Expected string value, got {type(value)}")

        if param_type == 'integer':
            try:
                return int(value.strip())
            except ValueError:
                raise ValueError(f"Cannot convert '{value}' to integer")

        elif param_type == 'number':
            try:
                return float(value.strip())
            except ValueError:
                raise ValueError(f"Cannot convert '{value}' to float")

        elif param_type == 'array':
            try:
                # Handle both bracketed and unbracketed formats
                if value.startswith('[') and value.endswith(']'):
                    array_str = value.strip('[]')
                else:
                    array_str = value

                if not array_str.strip():
                    return []

                return [int(x.strip()) for x in array_str.split(',')]
            except ValueError as e:
                raise ValueError(
                    f"Invalid array format or values: {value}. Error: {str(e)}")
        else:
            return str(value)

    except Exception as e:
        logger.error(
            f"Error converting parameter value: {value} to type {param_type}. Error: {str(e)}")
        raise ValueError(f"Parameter conversion failed: {str(e)}")


async def main():
    retry_count = 0

    # Configure logging for better debugging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    while retry_count < MAX_RETRIES:
        try:
            state.reset()  # Reset at the start of main
            logger.info("Starting main execution...")

            # Create a single MCP server connection
            logger.info("Establishing connection to MCP server...")
            server_params = StdioServerParameters(
                command="python",
                args=["example2-3.py", "dev"]
            )
            async with stdio_client(server_params) as (read, write):
                logger.info("Connection established, creating session...")
                async with ClientSession(read, write) as session:
                    logger.info("Session created, initializing...")
                    await session.initialize()

                    # Get available tools
                    logger.info("Requesting tool list...")
                    tools_result = await session.list_tools()
                    tools = tools_result.tools
                    logger.info(f"Successfully retrieved {len(tools)} tools")

                    # Create system prompt with available tools
                    print("Creating system prompt...")
                    print(f"Number of tools: {len(tools)}")

                    async def create_tools_description(tools):
                        """Create formatted description of available tools"""
                        try:
                            tools_description = []
                            for i, tool in enumerate(tools):
                                try:
                                    params = tool.inputSchema
                                    desc = getattr(
                                        tool, 'description', 'No description available')
                                    name = getattr(tool, 'name', f'tool_{i}')

                                    if 'properties' in params:
                                        param_details = []
                                        for param_name, param_info in params['properties'].items(
                                        ):
                                            param_type = param_info.get(
                                                'type', 'unknown')
                                            param_details.append(
                                                f"{param_name}: {param_type}")
                                        params_str = ', '.join(param_details)
                                    else:
                                        params_str = 'no parameters'

                                    tool_desc = f"{i+1}. {name}({params_str}) - {desc}"
                                    tools_description.append(tool_desc)
                                    logger.debug(
                                        f"Added description for tool: {tool_desc}")
                                except Exception as e:
                                    logger.error(
                                        f"Error processing tool {tool.name}: {str(e)}")
                                    tools_description.append(
                                        f"{i+1}. Error processing tool")

                            return "\n".join(tools_description)
                        except Exception as e:
                            logger.error(
                                f"Error creating tools description: {e}")
                            return "Error loading tools"

                    tools_description = await create_tools_description(tools)
                    if tools_description == "Error loading tools":
                        raise ValueError("Failed to create tools description")

                    logger.info("Successfully created tools description")

                    print("Created system prompt...")

                    system_prompt = f"""You are a math agent solving problems in iterations. You have access to various mathematical tools and PowerPoint functions.
                    Available tools:
                    {tools_description}

                        PowerPoint Functions:
                        1. open_powerpoint() - Opens a new PowerPoint presentation
                        2. draw_rectangle(x1: int, y1: int, x2: int, y2: int) - Draws a rectangle in PowerPoint (use values between 1-8 for coordinates)
                        3. add_text_in_powerpoint(text: str) - Adds text to PowerPoint
                        4. close_powerpoint() - Closes PowerPoint

                        You must follow this sequence for each problem:
                        1. First, perform all necessary mathematical calculations using FUNCTION_CALL
                        2. Then, use PowerPoint to visualize the results in this order:
                        - Open PowerPoint once at the start
                        - Draw a rectangle to highlight the results (use coordinates x1=2, y1=2, x2=7, y2=5 for center positioning)
                        - Add the Final Result inside the rectangle with this exact format:
                            "Final Result:\\n<calculated_value>"
                        - Close PowerPoint at the end

                        For array parameters, you can pass them in two formats:
                        1. As a comma-separated list: param1,param2,param3
                        2. As a bracketed list: [param1,param2,param3]

                        Example text format for PowerPoint:
                        POWERPOINT: add_text_in_powerpoint|Final Result:\\n7.59982224609308e+33

                        You must respond with EXACTLY ONE line in one of these formats (no additional text):
                        1. For function calls:
                        FUNCTION_CALL: function_name|param1|param2|...

                        2. For final answers:
                        FINAL_ANSWER: [7.59982224609308e+33]

                        3. For PowerPoint operations:
                        POWERPOINT: operation|param1|param2|...

                        Examples:
                        - FUNCTION_CALL: add|5|3
                        - FUNCTION_CALL: strings_to_chars_to_int|INDIA
                        - FUNCTION_CALL: int_list_to_exponential_sum|73,78,68,73,65
                        - POWERPOINT: open_powerpoint
                        - POWERPOINT: draw_rectangle|2|2|7|5
                        - POWERPOINT: add_text_in_powerpoint|Final Result:\\n7.59982224609308e+33
                        - POWERPOINT: close_powerpoint
                        - FINAL_ANSWER: [7.59982224609308e+33]

                        DO NOT include any explanations or additional text.
                        Your entire response should be a single line starting with either FUNCTION_CALL:, POWERPOINT:, or FINAL_ANSWER:"""

                    query = """Find the ASCII values of characters in INDIA and then return sum of exponentials of those values."""
                    logger.info("Starting iteration loop...")

                    while state.iteration < MAX_ITERATIONS:
                        logger.info(
                            f"\n--- Iteration {state.iteration + 1} ---")
                        current_query = query if state.last_response is None else f"{query}\n\n{' '.join(state.iteration_response)}"
                        current_query = current_query + "  What should I do next?"

                        # Get model's response with timeout
                        logger.info("Preparing to generate LLM response...")
                        prompt = f"{system_prompt}\n\nQuery: {current_query}"

                        try:
                            response = await generate_with_timeout(client, prompt)
                            response_text = response.text.strip()
                            logger.debug(f"Raw LLM Response: {response_text}")

                            # Process response to get valid command
                            for line in response_text.split('\n'):
                                line = line.strip()
                                if line.startswith(
                                        ("FUNCTION_CALL:", "FINAL_ANSWER:")):
                                    response_text = line
                                    logger.info(
                                        f"Processed response: {response_text}")
                                    break

                        except Exception as e:
                            print(f"Failed to get LLM response: {e}")
                            break

                        if response_text.startswith("FUNCTION_CALL:"):
                            _, function_info = response_text.split(":", 1)
                            parts = [p.strip()
                                        for p in function_info.split("|")]
                            func_name, params = parts[0], parts[1:]

                            print(
                                f"\nDEBUG: Raw function info: {function_info}")
                            print(f"DEBUG: Split parts: {parts}")
                            print(f"DEBUG: Function name: {func_name}")
                            print(f"DEBUG: Raw parameters: {params}")

                            try:
                                # Find the matching tool to get its input
                                # schema
                                tool = next(
                                    (t for t in tools if t.name == func_name), None)
                                if not tool:
                                    raise ValueError(
                                        f"Unknown tool: {func_name}")

                                arguments = {}
                                schema_properties = tool.inputSchema.get(
                                    'properties', {})

                                for param_name, param_info in schema_properties.items():
                                    if not params:
                                        raise ValueError(
                                            f"Not enough parameters for {func_name}")

                                    value = params.pop(0)
                                    param_type = param_info.get(
                                        'type', 'string')

                                    try:
                                        arguments[param_name] = await convert_param_value(value, param_type)
                                    except ValueError as e:
                                        logger.error(
                                            f"Error converting parameter {param_name}: {e}")
                                        raise ValueError(
                                            f"Invalid parameter {param_name}: {value}")

                                result = await session.call_tool(func_name, arguments=arguments)
                                print(f"DEBUG: Raw result: {result}")

                                # Get the full result content
                                if hasattr(result, 'content'):
                                    print(
                                        f"DEBUG: Result has content attribute")
                                    if isinstance(result.content, list):
                                        iteration_result = [
                                            str(item) for item in result.content]
                                    else:
                                        iteration_result = str(
                                            result.content)
                                else:
                                    print(
                                        f"DEBUG: Result has no content attribute")
                                    iteration_result = str(result)

                                print(
                                    f"DEBUG: Final iteration result: {iteration_result}")

                                # Format the response based on result type
                                if isinstance(iteration_result, list):
                                    result_str = f"[{', '.join(iteration_result)}]"
                                else:
                                    result_str = str(iteration_result)

                                state.iteration_response.append(
                                    f"In the {state.iteration + 1} iteration you called {func_name} with {arguments} parameters, "
                                    f"and the function returned {result_str}.")
                                state.last_response = iteration_result

                            except Exception as e:
                                print(f"DEBUG: Error details: {str(e)}")
                                print(f"DEBUG: Error type: {type(e)}")
                                import traceback
                                traceback.print_exc()
                                state.iteration_response.append(
                                    f"Error in iteration {state.iteration + 1}: {str(e)}")
                                break

                        elif response_text.startswith("POWERPOINT:"):
                            try:
                                _, powerpoint_info = response_text.split(
                                    ":", 1)
                                parts = [
                                    p.strip() for p in powerpoint_info.split("|")]
                                operation, params = parts[0], parts[1:]

                                logger.info(
                                    f"Processing PowerPoint operation: {operation} with params: {params}")

                                # Map PowerPoint operations to actual tool
                                # functions
                                if operation == "open_powerpoint":
                                    result = await session.call_tool("open_powerpoint", arguments={})
                                elif operation == "close_powerpoint":
                                    result = await session.call_tool("close_powerpoint", arguments={})
                                elif operation == "draw_rectangle":
                                    if len(params) < 4:
                                        raise ValueError(
                                            "Not enough parameters for draw_rectangle")
                                    arguments = {
                                        "x1": await convert_param_value(params[0], "integer"),
                                        "y1": await convert_param_value(params[1], "integer"),
                                        "x2": await convert_param_value(params[2], "integer"),
                                        "y2": await convert_param_value(params[3], "integer")
                                    }
                                    result = await session.call_tool("draw_rectangle", arguments=arguments)
                                elif operation == "add_text_in_powerpoint":
                                    if not params:
                                        raise ValueError(
                                            "Text parameter is required for add_text_in_powerpoint")
                                    result = await session.call_tool("add_text_in_powerpoint", arguments={"text": params[0]})
                                else:
                                    raise ValueError(
                                        f"Unknown PowerPoint operation: {operation}")

                                # Format the response
                                if hasattr(result, 'content'):
                                    if isinstance(result.content, list):
                                        result_str = ", ".join(
                                            [str(item) for item in result.content])
                                    else:
                                        result_str = str(result.content)
                                else:
                                    result_str = str(result)

                                state.iteration_response.append(
                                    f"In the {state.iteration + 1} iteration you performed PowerPoint operation '{operation}', "
                                    f"and the result was: {result_str}.")

                            except Exception as e:
                                logger.error(
                                    f"Error in PowerPoint operation: {e}")
                                state.iteration_response.append(
                                    f"Error in PowerPoint operation: {str(e)}")
                                continue

                        elif response_text.startswith("FINAL_ANSWER:"):
                            try:
                                _, answer = response_text.split(":", 1)
                                answer = answer.strip()
                                if answer.startswith(
                                        "[") and answer.endswith("]"):
                                    final_result = answer[1:-1].strip()
                                    logger.info(
                                        f"Final result: {final_result}")
                                    state.iteration_response.append(
                                        f"Final result: {final_result}")
                                    break
                                else:
                                    logger.warning(
                                        "Invalid final answer format")
                                    continue
                            except Exception as e:
                                logger.error(
                                    f"Error processing final answer: {e}")
                                continue

            state.iteration += 1

            break  # If we get here, everything worked fine

        except KeyboardInterrupt:
            logger.warning("\nKeyboard interrupt detected, cleaning up...")
            state.reset()
            break
        except Exception as e:
            logger.error(f"Main execution error: {e}")
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                logger.error("Max retries exceeded. Exiting.")
                break
            logger.info(f"Retrying... Attempt {retry_count + 1}")
            if retry_count < MAX_RETRIES:
                logger.info(f"Retrying in 5 seconds...")
                await asyncio.sleep(5)
            else:
                logger.error("Max retries reached, exiting...")
                raise
        finally:
            state.reset()  # Reset at the end of main

if __name__ == "__main__":
    try:
        # Configure and run event loop
        if sys.platform == 'win32':
            loop = asyncio.ProactorEventLoop()
        else:
            loop = asyncio.new_event_loop()

        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(main())
        except KeyboardInterrupt:
            logger.warning("\nExiting due to keyboard interrupt...")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
            logger.error("Traceback:", exc_info=True)
        finally:
            # Ensure all tasks are complete and close the loop
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()

            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )

            loop.close()

    except Exception as e:
        logger.critical(f"Failed to initialize event loop: {e}")
        sys.exit(1)
