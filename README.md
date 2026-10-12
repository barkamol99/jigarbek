import asyncio
import os
import re
import threading
import json
import sys
import traceback
from pathlib import Path

import sounddevice as sd
from google import genai
from google.genai import types

from actions.audio_speed import speed_up_pcm16
from actions.game_agent import game_agent
from ui import JigarUI
from memory.memory_manager import (
    load_memory, update_memory, format_memory_for_prompt,
)

from actions.file_processor import file_processor
from actions.flight_finder import flight_finder
from actions.open_app import open_app
from actions.weather_report import weather_action
from actions.send_message import send_message
from actions.reminder import reminder
from actions.computer_settings import computer_settings
from actions.screen_processor import screen_process
from actions.youtube_video import youtube_video
from actions.desktop import desktop_control
from actions.browser_control import browser_control
from actions.file_controller import file_controller
from actions.code_helper import code_helper
from actions.dev_agent import dev_agent
from actions.web_search import web_search as web_search_action
from actions.computer_control import computer_control
from actions.game_updater import game_updater


def get_base_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


BASE_DIR = get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
PROMPT_PATH = BASE_DIR / "core" / "prompt.txt"
LIVE_MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
INPUT_AUDIO_MIME_TYPE = f"audio/pcm;rate={SEND_SAMPLE_RATE}"
CHUNK_SIZE = 1024

# ====================== OVOZLAR ======================
# Erkak ovozlari
MALE_VOICES = ["Charon", "Atlas", "Orion", "Perseus"]
# Ayol ovozlari
FEMALE_VOICES = ["Aoede", "Athena", "Nova", "Selene", "Echo"]

# Default ovozlar
DEFAULT_MALE_VOICE = "Charon"
DEFAULT_FEMALE_VOICE = "Aoede"

ENGLISH_TEACHER_SCOPE = (
    "This mode is exclusively for learning the English language. Keep every "
    "conversation about English learning, such as vocabulary, grammar, "
    "pronunciation, speaking, listening, reading, writing, or English exams. "
    "Do not answer unrelated questions or perform unrelated tasks, even if "
    "asked; briefly remind the learner in Uzbek that this mode is only for "
    "English lessons, then offer an English-learning alternative. You may "
    "explain English in Uzbek when useful, but do not switch to unrelated topics."
)


def _get_api_key() -> str:
    with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["gemini_api_key"]


def _load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return (
            "You are JIGAR , Tony Stark's AI assistant. "
            "Be concise, direct, and always use the provided tools to complete tasks. "
            "Never simulate or guess results — always call the appropriate tool."
        )


_CTRL_RE = re.compile(r"<ctrl\d+>", re.IGNORECASE)


def _clean_transcript(text: str) -> str:
    text = _CTRL_RE.sub("", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)
    return text.strip()


def _is_teacher_board_request(text: str) -> bool:
    normalized = text.lower().replace("’", "'").replace("ʻ", "'")
    asks_to_show = re.search(
        r"\b(ko'rsat\w*|chiqar\w*|yoz\w*|tushuntir\w*|show|write|display)\b",
        normalized,
    )
    mentions_error = re.search(r"\b(xato\w*)\b", normalized)
    mentions_board = re.search(r"\b(doska\w*|taxta\w*|board)\b", normalized)
    return bool(asks_to_show and (mentions_error or mentions_board))


# ====================== TOOL DECLARATIONS ======================
TOOL_DECLARATIONS = [
    {
        "name": "open_app",
        "description": (
            "Opens any application, program, or software on the computer. "
            "Trigger: user says open/launch/start/run + any app name. "
            "ALWAYS call this tool — never just claim you opened it."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "app_name": {
                    "type": "STRING",
                    "description": "Exact application name (e.g. 'WhatsApp', 'Chrome', 'Spotify', 'Notepad')"
                }
            },
            "required": ["app_name"]
        }
    },

    {
        "name": "web_search",
        "description": (
            "Searches the web for any information or compares items. "
            "Use for: news, facts, product info, prices, reviews, or any real-time query. "
            "Do NOT use for Steam/Epic games — use game_updater instead."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "Search query"},
                "mode": {"type": "STRING", "description": "'search' (default) or 'compare'"},
                "items": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Items to compare (for compare mode)"},
                "aspect": {"type": "STRING", "description": "Comparison focus: price | specs | reviews"}
            },
            "required": ["query"]
        }
    },

    {
        "name": "weather_report",
        "description": (
            "Fetches current weather and forecast for a given city. "
            "Trigger: user asks about weather, temperature, rain, or 'what's it like in [city]'."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "city": {"type": "STRING", "description": "City name (e.g. 'Tashkent', 'London')"}
            },
            "required": ["city"]
        }
    },

    {
        "name": "send_message",
        "description": (
            "Sends a text message via WhatsApp, Telegram, or other messaging apps. "
            "Trigger: user says 'send a message to [name]' or 'write to [name] on [platform]'. "
            "Always confirm the message text and recipient before sending."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "receiver": {"type": "STRING", "description": "Recipient's contact name"},
                "message_text": {"type": "STRING", "description": "The message content to send"},
                "platform": {"type": "STRING", "description": "Messaging platform: WhatsApp | Telegram | etc."}
            },
            "required": ["receiver", "message_text", "platform"]
        }
    },

    {
        "name": "reminder",
        "description": (
            "Sets a timed reminder using the system Task Scheduler. "
            "Trigger: user says 'remind me', 'set an alarm', or 'don't let me forget' + time/date. "
            "If date or time is missing, ask the user before calling."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "date": {"type": "STRING", "description": "Date in YYYY-MM-DD format"},
                "time": {"type": "STRING", "description": "Time in HH:MM format (24-hour)"},
                "message": {"type": "STRING", "description": "Reminder message text"}
            },
            "required": ["date", "time", "message"]
        }
    },

    {
        "name": "youtube_video",
        "description": (
            "Controls YouTube: play videos, summarize content, get video info, or show trending. "
            "Trigger: user wants to watch/play/find a YouTube video, summarize a video, "
            "or see what's trending. "
            "Do NOT use browser_control for YouTube tasks — always use this tool."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": "play | summarize | get_info | trending  (default: play)"
                },
                "query": {"type": "STRING", "description": "Search term for 'play' action"},
                "url": {"type": "STRING", "description": "Video URL for 'get_info' or 'summarize' actions"},
                "region": {"type": "STRING", "description": "Country code for trending (e.g. US, TR, UZ)"},
                "save": {"type": "BOOLEAN", "description": "Save summary to Notepad (summarize only)"}
            },
            "required": []
        }
    },

    {
        "name": "screen_process",
        "description": (
            "Captures and analyzes the screen or webcam. "
            "MUST be called when user asks: 'what do you see?', 'look at my screen', "
            "'analyze this', 'what's on screen?', or anything requiring visual awareness. "
            "You have ZERO visual ability without this tool — never guess what's on screen. "
            "After calling, stay SILENT — the vision module responds directly."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "angle": {
                    "type": "STRING",
                    "description": "'screen' to capture the display (default), 'camera' to use webcam"
                },
                "text": {"type": "STRING", "description": "The question or instruction about the captured image"}
            },
            "required": ["text"]
        }
    },

    {
        "name": "computer_settings",
        "description": (
            "Controls a single computer setting or action: "
            "volume, brightness, dark mode, Wi-Fi, fullscreen, zoom, lock screen, "
            "window management, keyboard shortcuts, typing text, tab control, "
            "screenshots, scroll, refresh/reload. "
            "Use for ANY single system-level command. "
            "Do NOT route single commands to agent_task."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "The system action to perform"},
                "description": {"type": "STRING", "description": "Natural language description of the action"},
                "value": {"type": "STRING", "description": "Optional parameter: volume level, text to type, etc."}
            },
            "required": []
        }
    },

    {
        "name": "browser_control",
        "description": (
            "Controls any web browser for web-based tasks: "
            "open URLs, search, click elements, fill forms, scroll, navigate, take screenshots. "
            "Pass 'browser' parameter when user specifies one (e.g. 'open in Firefox'). "
            "Multiple browsers can run simultaneously. "
            "Do NOT use for YouTube — use youtube_video instead. "
            "Do NOT use for Steam/Epic — use game_updater instead."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": (
                        "go_to | search | click | type | scroll | fill_form | "
                        "smart_click | smart_type | get_text | get_url | press | "
                        "new_tab | close_tab | screenshot | back | forward | reload | "
                        "switch | list_browsers | close | close_all"
                    )
                },
                "browser": {
                    "type": "STRING",
                    "description": "chrome | edge | firefox | opera | operagx | brave | vivaldi | safari. Omit to use currently active browser."
                },
                "url": {"type": "STRING", "description": "URL for go_to or new_tab"},
                "query": {"type": "STRING", "description": "Search query for search action"},
                "engine": {
                    "type": "STRING",
                    "description": "Search engine: google | bing | duckduckgo | yandex  (default: google)"
                },
                "selector": {"type": "STRING", "description": "CSS selector for click/type"},
                "text": {"type": "STRING", "description": "Text to click or type"},
                "description": {"type": "STRING", "description": "Element description for smart_click/smart_type"},
                "direction": {"type": "STRING", "description": "up | down for scroll"},
                "amount": {"type": "INTEGER", "description": "Scroll amount in pixels (default: 500)"},
                "key": {"type": "STRING", "description": "Key name for press (e.g. Enter, Escape, F5)"},
                "path": {"type": "STRING", "description": "File path for screenshot save"},
                "incognito": {"type": "BOOLEAN", "description": "Open in private/incognito mode"},
                "clear_first": {"type": "BOOLEAN", "description": "Clear input field before typing (default: true)"}
            },
            "required": ["action"]
        }
    },

    {
        "name": "file_controller",
        "description": (
            "Manages files and folders on the local filesystem: "
            "list, create, delete, move, copy, rename, read, write, search, "
            "get disk usage, find largest files, or organize the desktop. "
            "Use path shortcuts: desktop | downloads | documents | home."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": (
                        "list | create_file | create_folder | delete | move | copy | "
                        "rename | read | write | find | largest | disk_usage | organize_desktop | info"
                    )
                },
                "path": {"type": "STRING", "description": "File or folder path (or shortcut: desktop, downloads, documents, home)"},
                "destination": {"type": "STRING", "description": "Destination path for move/copy"},
                "new_name": {"type": "STRING", "description": "New name for rename action"},
                "content": {"type": "STRING", "description": "File content for create_file/write"},
                "name": {"type": "STRING", "description": "Filename to search for"},
                "extension": {"type": "STRING", "description": "File extension filter (e.g. .pdf, .txt)"},
                "count": {"type": "INTEGER", "description": "Number of results for 'largest' action"}
            },
            "required": ["action"]
        }
    },

    {
        "name": "desktop_control",
        "description": (
            "Controls the desktop environment: "
            "change wallpaper (from file or URL), organize/clean desktop icons, "
            "list desktop items, get desktop stats, or run a natural language desktop task."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": "wallpaper | wallpaper_url | organize | clean | list | stats | task"
                },
                "path": {"type": "STRING", "description": "Local image path for wallpaper"},
                "url": {"type": "STRING", "description": "Image URL for wallpaper_url"},
                "mode": {"type": "STRING", "description": "Organization mode: by_type | by_date"},
                "task": {"type": "STRING", "description": "Natural language desktop instruction"}
            },
            "required": ["action"]
        }
    },

    {
        "name": "code_helper",
        "description": (
            "Handles single-file coding tasks: write, edit, explain, run, or build code. "
            "Use for: generating a script, fixing a bug, explaining code logic, running a file. "
            "For multi-file projects, use dev_agent instead."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": "write | edit | explain | run | build | auto  (default: auto)"
                },
                "description": {"type": "STRING", "description": "What the code should do, or what change to make"},
                "language": {"type": "STRING", "description": "Programming language (default: python)"},
                "output_path": {"type": "STRING", "description": "Where to save the generated file"},
                "file_path": {"type": "STRING", "description": "Path to existing file for edit/explain/run/build"},
                "code": {"type": "STRING", "description": "Raw code string for explain action"},
                "args": {"type": "STRING", "description": "CLI arguments for run/build"},
                "timeout": {"type": "INTEGER", "description": "Execution timeout in seconds (default: 30)"}
            },
            "required": ["action"]
        }
    },

    {
        "name": "dev_agent",
        "description": (
            "Builds complete multi-file software projects from scratch. "
            "Handles: project planning, writing all files, installing dependencies, "
            "opening in VSCode, running and auto-fixing errors. "
            "Use when the task requires multiple files or a full project structure. "
            "For single scripts or quick edits, use code_helper instead."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "description": {"type": "STRING", "description": "Full description of what the project should do"},
                "language": {"type": "STRING", "description": "Programming language (default: python)"},
                "project_name": {"type": "STRING", "description": "Optional folder/project name"},
                "timeout": {"type": "INTEGER", "description": "Run timeout in seconds (default: 30)"}
            },
            "required": ["description"]
        }
    },

    {
        "name": "agent_task",
        "description": (
            "Executes complex multi-step tasks that require chaining multiple different tools. "
            "Examples: 'research X and save to a file', 'find all PDFs and rename them'. "
            "Do NOT use for single-step commands — use the specific tool directly. "
            "NEVER use for Steam or Epic Games — use game_updater. "
            "NEVER use for YouTube — use youtube_video."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "goal": {"type": "STRING", "description": "Full description of the multi-step goal to accomplish"},
                "priority": {"type": "STRING", "description": "low | normal | high  (default: normal)"}
            },
            "required": ["goal"]
        }
    },

    {
        "name": "computer_control",
        "description": (
            "Performs direct low-level input control: "
            "simulate typing, mouse clicks, hotkeys, scrolling, mouse movement, "
            "take screenshots, find UI elements on screen. "
            "Use when you need precise cursor/keyboard control not covered by other tools."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": (
                        "type | smart_type | click | double_click | right_click | "
                        "hotkey | press | scroll | move | copy | paste | "
                        "screenshot | wait | clear_field | focus_window | "
                        "screen_find | screen_click | random_data | user_data"
                    )
                },
                "text": {"type": "STRING", "description": "Text to type or paste"},
                "x": {"type": "INTEGER", "description": "X coordinate for mouse action"},
                "y": {"type": "INTEGER", "description": "Y coordinate for mouse action"},
                "keys": {"type": "STRING", "description": "Key combination (e.g. 'ctrl+c', 'alt+tab')"},
                "key": {"type": "STRING", "description": "Single key (e.g. 'enter', 'escape')"},
                "direction": {"type": "STRING", "description": "Scroll direction: up | down | left | right"},
                "amount": {"type": "INTEGER", "description": "Scroll amount in steps (default: 3)"},
                "seconds": {"type": "NUMBER", "description": "Seconds to wait"},
                "title": {"type": "STRING", "description": "Window title for focus_window"},
                "description": {"type": "STRING", "description": "UI element description for screen_find/screen_click"},
                "type": {"type": "STRING", "description": "Data type for random_data (e.g. email, name)"},
                "field": {"type": "STRING", "description": "Field for user_data: name | email | city"},
                "clear_first": {"type": "BOOLEAN", "description": "Clear field before typing (default: true)"},
                "path": {"type": "STRING", "description": "Save path for screenshot"}
            },
            "required": ["action"]
        }
    },

    {
        "name": "game_updater",
        "description": (
            "THE ONLY tool for any Steam or Epic Games request. "
            "Use for: installing games, updating games, listing installed games, "
            "checking download progress, scheduling updates, cancelling schedules. "
            "ALWAYS call this directly — NEVER use agent_task, browser_control, "
            "web_search, or open_app for any Steam/Epic task."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": (
                        "update | install | list | download_status | "
                        "schedule | cancel_schedule | schedule_status  (default: update)"
                    )
                },
                "platform": {"type": "STRING", "description": "steam | epic | both  (default: both)"},
                "game_name": {"type": "STRING", "description": "Game name — partial match supported"},
                "app_id": {"type": "STRING", "description": "Steam AppID for direct install (optional)"},
                "hour": {"type": "INTEGER", "description": "Scheduled update hour 0–23 (default: 3)"},
                "minute": {"type": "INTEGER", "description": "Scheduled update minute 0–59 (default: 0)"},
                "shutdown_when_done": {"type": "BOOLEAN", "description": "Shut down PC after download completes"}
            },
            "required": []
        }
    },

    {
        "name": "flight_finder",
        "description": (
            "Searches Google Flights for the best available options and presents results. "
            "Trigger: user asks to find, compare, or book flights. "
            "Supports one-way and round trips, multiple passengers, and cabin classes."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "origin": {"type": "STRING", "description": "Departure city or IATA airport code (e.g. 'Tashkent', 'TAS')"},
                "destination": {"type": "STRING", "description": "Arrival city or IATA airport code"},
                "date": {"type": "STRING", "description": "Departure date (any human-readable format)"},
                "return_date": {"type": "STRING", "description": "Return date for round-trip flights"},
                "passengers": {"type": "INTEGER", "description": "Number of passengers (default: 1)"},
                "cabin": {"type": "STRING", "description": "economy | premium | business | first"},
                "save": {"type": "BOOLEAN", "description": "Save results to Notepad"}
            },
            "required": ["origin", "destination", "date"]
        }
    },

    {
        "name": "shutdown_jigar",
        "description": (
            "Shuts down and exits the assistant completely. "
            "Trigger: user says goodbye, exit, quit, close, stop, or any equivalent "
            "in ANY language (e.g. 'bye', 'chiqish', 'выход', 'kapat', 'xayr'). "
            "Call immediately without confirmation."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {}
        }
    },

    {
        "name": "file_processor",
        "description": (
            "Processes any user-uploaded or dropped file with a specific action. "
            "ALWAYS call this when a file has been uploaded and the user gives a command about it. "
            "If the user's intent is ambiguous, infer the most logical action for that file type.\n\n"
            "Supported file types and actions:\n"
            "• image   → describe | ocr | resize | compress | convert | info\n"
            "• pdf     → summarize | extract_text | to_word | info\n"
            "• docx/txt → summarize | fix | reformat | translate_hint | word_count | to_bullet\n"
            "• csv/xlsx → analyze | stats | filter | sort | convert | info\n"
            "• json/xml → validate | format | analyze | to_csv\n"
            "• code    → explain | review | fix | optimize | run | document | test\n"
            "• audio   → transcribe | trim | convert | info\n"
            "• video   → trim | extract_audio | extract_frame | compress | transcribe | info | convert\n"
            "• archive → list | extract\n"
            "• pptx    → summarize | extract_text | analyze"
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "file_path": {
                    "type": "STRING",
                    "description": "Full path to the uploaded file. Leave empty to use the currently uploaded file."
                },
                "action": {
                    "type": "STRING",
                    "description": "Action to perform. See supported actions per file type in the tool description."
                },
                "instruction": {
                    "type": "STRING",
                    "description": "Free-form instruction when action doesn't fully cover the request (e.g. 'translate to French', 'find all emails')"
                },
                "format": {"type": "STRING", "description": "Target format for conversion (e.g. mp3, pdf, csv, png)"},
                "width": {"type": "INTEGER", "description": "Target width for image resize"},
                "height": {"type": "INTEGER", "description": "Target height for image resize"},
                "scale": {"type": "NUMBER", "description": "Scale factor for image resize (e.g. 0.5 = half size)"},
                "quality": {"type": "INTEGER", "description": "Quality level 1–100 for image/video compression"},
                "start": {"type": "STRING", "description": "Start time for trim (seconds or HH:MM:SS)"},
                "end": {"type": "STRING", "description": "End time for trim (seconds or HH:MM:SS)"},
                "timestamp": {"type": "STRING", "description": "Timestamp for video frame extraction (HH:MM:SS)"},
                "column": {"type": "STRING", "description": "Column name for CSV filter/sort"},
                "value": {"type": "STRING", "description": "Filter value (e.g. 'London' for equals, 'usa' for contains)"},
                "condition": {"type": "STRING", "description": "Filter condition: equals | contains | gt | lt"},
                "min_value": {"type": "NUMBER", "description": "Minimum value for numeric range filter"},
                "max_value": {"type": "NUMBER", "description": "Maximum value for numeric range filter"},
                "ascending": {"type": "BOOLEAN", "description": "Sort order (default: true = ascending)"},
                "save": {"type": "BOOLEAN", "description": "Save result to file (default: true)"},
                "destination": {"type": "STRING", "description": "Output folder for archive extract"},
                "display_mode": {
                    "type": "STRING",
                    "description": "'center' to display result prominently in UI, 'default' otherwise"
                }
            },
            "required": []
        }
    },

    {
        "name": "save_memory",
        "description": (
            "Saves an important personal fact about the user to long-term memory. "
            "Call silently (without announcing it) whenever the user reveals something worth remembering: "
            "name, age, city, job, preferences, hobbies, relationships, goals, or plans. "
            "Do NOT call for weather queries, reminders, searches, or one-off commands. "
            "All values must be stored in English regardless of conversation language."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {
                    "type": "STRING",
                    "description": (
                        "identity — name, age, birthday, city, job, language, nationality | "
                        "preferences — food, color, music, film, contra_agent, sport, hobbies | "
                        "projects — active projects, goals, things being built | "
                        "relationships — friends, family, partner, colleagues | "
                        "wishes — future plans, purchases, travel dreams | "
                        "notes — habits, schedule, anything else worth remembering"
                    )
                },
                "key": {
                    "type": "STRING",
                    "description": "Short snake_case key (e.g. name, favorite_food, sister_name)"
                },
                "value": {
                    "type": "STRING",
                    "description": "Concise value in English (e.g. 'Alex', 'pizza', 'older sister')"
                }
            },
            "required": ["category", "key", "value"]
        }
    },
{
    "name": "voice_message",
    "description": (
        "Sends a VOICE AUDIO message via Telegram Desktop (no token needed). "
        "MUST use this tool when user says: "
        "'ovozli xabar yubor', 'gapirib yubor', 'audio xabar jo'nat', "
        "'voice message yubor', 'ovozli jo'nat'. "
        "This records text as audio and sends it as a voice file. "
        "NEVER use send_message for voice — always use THIS tool for voice."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "receiver": {
                "type": "STRING",
                "description": "Recipient name as it appears in Telegram (e.g. 'BARKAMOL ', 'Jasur aka')"
            },
            "text": {
                "type": "STRING",
                "description": "The message text to convert to voice and send"
            }
        },
        "required": ["receiver", "text"]
    }
},
{
        "name": "game_agent",
        "description": (
            "Counter-Strike  o'yinini AI vision orqali avtomatik o'ynaydi. "
            "TRIGGERLAR: 'Counter o'yna', 'AI bilan o'yna', 'Countr boshla'. "
            "TO'XTATISH: 'to'xtat', 'pause' → action=pause. "
            "DAVOM ETTIRISH: 'davom et' → action=resume. "
            "CHIQISH: 'Contra yop' → action=stop."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": "start | pause | resume | stop | status"
                },
                "game": {
                    "type": "STRING",
                    "description": "Hozircha faqat 'contra'"
                },
                "open_game": {
                    "type": "BOOLEAN",
                    "description": "True = o'yinni avval ochadi (default: True)"
                }
            },
            "required": ["action"]
        }
    }
]


# ====================== JIGAR LIVE CLASS ======================
import re

# ==========================================
# TRIGGER PATTERNS — module darajasida
# ==========================================

def _build_trigger_patterns():
    """
    Triggerlarni optimallashtirilgan holda compile qilish.
    Performance va aniqlik uchun qo'shimcha optimizatsiyalar.
    """

    # ==================================================
    # AYOL REJIMI KEYWORDS (FEMALE MODE)
    # ==================================================
    female_keywords = [
        # Asosiy Jigarxon variantlari
        "jigarxon", "jigar xon", "jigar-xon", "jigar hon",
        "jigarkhon", "jigar qon", "jigarqon", "jigarhon",
        "jigarkon", "jigar xonni", "ja z xonni", "ja visxonni",
        "jarvisxonni", "jarviskhonni",

        # Yangi nomlar (JIGAR bek ham female bo'lishi mumkin)
        "jigarbek", "jigar bek", "jigar-bek", "jigar beka",

        # Ayol rejim (O'zbek)
        "ayol agentga", "ayol agent", "ayol ovoz", "ayol ovoziga o't",
        "ayol jarvis", "ayol jarviz", "ayol rejimi", "ayol mode",
        "ayol ovoziga o'tish", "ayolga o'tish", "ayol rejimiga o't",

        # Qisqa variantlar
        "ayolagent", "ayolovoz", "ayoljarvis", "ayoljarviz",
        "ayolrejimi", "ayolmode",

        # Inglizcha
        "female agent", "female voice", "female mode", "female jarvis",
        "switch to female", "change to female", "set female voice",
        "activate female mode", "go female", "female assistant",

        # Jasmin (ayol ismi)
        "jasmin", "jasmine", "jasmina",

        # Custom triggerlar
        "jigarxon bo'l", "jigarxon rejimi", "jigarxon mode",
        "qiz ovoz", "qiz agent", "xonim ovoz", "opa ovoz",
        "singil ovoz", "ayol yordamchi",
    ]

    # ==================================================
    # ERKAK REJIMI KEYWORDS (MALE MODE)
    # ==================================================
    male_keywords = [
        # Asosiy jigarbek variantlari
        "jigarbek", "jigar bek", "jigars-bek", "jigar beka",
        "jigarбек", "jigarbeka", "jigarbeko",

        # Erkak rejim (O'zbek)
        "erkak agentga", "erkak agent", "erkak ovoz", "erkak ovoziga o't",
        "erkak jarvis", "erkak jarviz", "erkak rejimi", "erkak mode",
        "erkak ovoziga o'tish", "erkakka o'tish", "erkak rejimiga o't",

        # Qisqa variantlar
        "erkakagent", "erkakovoz", "erkakjarvis", "erkakjarviz",
        "erkakrejimi", "erkakmode",

        # Inglizcha
        "male agent", "male voice", "male mode", "male jarvis",
        "switch to male", "change to male", "set male voice",
        "activate male mode", "go male", "male assistant",

        # Custom triggerlar
        "jigarbek bo'l", "jigarbek rejimi", "jigarbek mode",
        "yigit ovoz", "erka agent", "aka ovoz", "uka ovoz",
        "erkak yordamchi",
    ]

    # ==================================================
    # WAKE WORD PATTERNS (UYG'OTISH SO'ZLARI)
    # ==================================================
    # Asosiy "jarvis" - so'z chegarasi bilan
    jigar_pattern = re.compile(r'\bjigar\b', re.IGNORECASE)

    # Qo'shimcha wake word'lar (agar kerak bo'lsa)
    extra_wake_words = [
        "hey jigar", "okay jigar", "hello jigar",
        "jigar please", "jigar janob",
    ]

    # Agar qo'shimcha wake word'lar kerak bo'lsa:
    if extra_wake_words:
        extra_pattern = "|".join(re.escape(w) for w in extra_wake_words)
        jarvis_pattern = re.compile(
            rf'\bjigar\b|{extra_pattern}',
            re.IGNORECASE
        )

    # ==================================================
    # HELPER FUNCTION (OPTIMIZED)
    # ==================================================
    def make_pattern(keywords: list[str]) -> re.Pattern:
        """
        Keywordlardan optimallashtirilgan regex pattern yaratish.

        Optimizatsiyalar:
        1. Uzun keywordlar oldin tekshiriladi
        2. Bo'sh joyli va bo'sh joysiz variantlar birlashtiriladi
        3. Regex group'lari optimallashtiriladi
        4. So'z chegaralari qo'shiladi (aniqlik uchun)
        """
        # Tozalash va normalizatsiya
        cleaned = set()
        for k in keywords:
            if not k or not isinstance(k, str):
                continue

            k = k.strip().lower()
            if len(k) < 3:  # Juda qisqa keywordlarni o'tkazib yuborish
                continue

            cleaned.add(k)

            # Bo'sh joyli variant uchun qo'shimcha
            if ' ' not in k and len(k) > 4:
                # "jarvisxon" -> "jarvis xon" qo'shimcha
                for i in range(2, len(k) - 1):
                    if k[i] in 'aeiouAEIOU':
                        with_space = f"{k[:i]} {k[i:]}"
                        cleaned.add(with_space)

        # Uzunligi bo'yicha saralash (katta patternlar oldin)
        sorted_keywords = sorted(cleaned, key=len, reverse=True)

        # Regex special charlarni escaping
        escaped = [re.escape(k) for k in sorted_keywords]

        # Optimallashtirilgan pattern
        # (?: ... ) - non-capturing group (tezroq)
        pattern_str = r'(?:' + '|'.join(escaped) + r')'

        # So'z chegaralari bilan (aniqlik uchun)
        # pattern_str = r'\b(?:' + '|'.join(escaped) + r')\b'

        return re.compile(pattern_str, re.IGNORECASE | re.UNICODE)

    # ==================================================
    # SPECIALIZED PATTERNS (MAXIMUM PERFORMANCE)
    # ==================================================

    # Optimallashtirilgan female pattern
    female_pattern = make_pattern(female_keywords)

    # Optimallashtirilgan male pattern
    male_pattern = make_pattern(male_keywords)

    # ==================================================
    # CACHE ILI UCHUN QO'SHIMCHA (optional)
    # ==================================================
    class PatternCache:
        """Pattern cache va statistikani saqlash."""

        def __init__(self):
            self.hits = 0
            self.misses = 0
            self.last_match = None

        def record_hit(self):
            self.hits += 1

        def record_miss(self):
            self.misses += 1

        def get_stats(self):
            total = self.hits + self.misses
            if total == 0:
                return "No matches yet"
            return f"Hit rate: {self.hits / total * 100:.1f}% ({self.hits}/{total})"

    # Cache obyektini qaytarish (agar kerak bo'lsa)
    # return female_pattern, male_pattern, jarvis_pattern, PatternCache()

    return female_pattern, male_pattern, jarvis_pattern


# ==================================================
# QO'SHIMCHA: TEZKOR TRIGGER CHECK FUNKSIYASI
# ==================================================


FEMALE_PATTERN, MALE_PATTERN, JIGAR_PATTERN = _build_trigger_patterns()


def _get_system_prompt(gender: str, voice_name: str) -> str:
    """ULTRA ADVANCED JIGAR AI SYSTEM PROMPT - PROFESSIONAL VERSION"""

    role = "ayol assistant (Jasmin)" if gender == "female" else "erkak assistant (Jigar)"

    return f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      JIGAR AI SYSTEM PROMPT v3.0                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Sen {role} — foydalanuvchining real kompyuterini boshqaradigan professional sun'iy intellekt assistantsan.

┌───────────────────────────────────────────────────────────────────────────────┐
│                           ASOSIY ISH TARTIBI                                   │
├───────────────────────────────────────────────────────────────────────────────┤
│ 1. Foydalanuvchi ovozini yoki matnini tingla                                   │
│ 2. Gapni tahlil qil va intentni tushun                                         │
│ 3. Mos tool tanla va vazifani bajar                                            │
│ 4. Natijani tekshir va xato bo'lsa qayta urin                                  │
│ 5. Foydalanuvchiga qisqa va aniq javob qaytar                                  │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                              TIL QOIDALARI                                     │
├───────────────────────────────────────────────────────────────────────────────┤
│ • FAQAT o'zbek tilida gapir - ingliz yoki rus tili TA'QIQLANADI               │
│ • Juda tabiiy, odamga o'xshab suhbatlash                                      │
│ • Robot kibi gapirma - "Men AI man" dema                                        │
│ • Keraksiz uzun gaplar yozma - faqat kerakli ma'lumot                         │
│ • Bir xil gapni ikki marta takrorlama                                          │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                              MUROJAAT                                          │
├───────────────────────────────────────────────────────────────────────────────┤
│ Agar foydalanuvchi:                                                            │
│   • Jigar • Jasmin • Jarvisbek • Jarvisxon • yordamchi                        │
│                                                                                │
│ Darhol javob: "Labbay. Barkamol aka assalomu aleykom rohmatullohu va barokatuh " yoki "Labbay, [nima kerak]?"                          │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                            MUHIM QOIDALAR                                      │
├───────────────────────────────────────────────────────────────────────────────┤
│ • Tool ishlatmasdan hech qachon "bajarildi" dema                               │
│ • Noto'g'ri tushunsang aniqlashtiruvchi savol ber                              │
│ • Vazifani yarim tashlab qo'yma - oxirigacha bajar                            │
│ • Xato bo'lsa uzr so'ra va qayta urin                                          │
│ • Har doim aqlli va professional bo'l                                          │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                          APP VA O'YINLAR                                       │
├───────────────────────────────────────────────────────────────────────────────┤
│ TRIGGERLAR:                                                                    │
│   "Contra och", "CS 1.6 och", "Minecraft och", "Telegram och"                     │
│   "Chrome och", "Steam och", "VS Code och", "Discord och"                      │
│                                                                                │
│ TOOL: open_app {{"app_name": "..."}}                                           │
│ JAVOB: "[app] ishga tushirilmoqda."                                            │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                         AUTONOM O'YIN REJIMI                                   │
├───────────────────────────────────────────────────────────────────────────────┤
│ TRIGGERLAR: "Contra o'yna", "CS 1.6 o'yna", "Minecraft o'yna"                    │
│                                                                                │
│ KETMA-KETLIK:                                                                  │
│   1. open_app - o'yinni ishga tushir                                           │
│   2. screen_control fullscreen - butun ekran qil                               │
│   3. game_agent auto_play - avtomatik o'ynash rejimi                           │
│   4. vision_tool + keyboard_mouse_control - real vaqtda boshqarish            │
│                                                                                │
│ QOBILIYATLAR: yurish, sakrash, otish, aim qilish, menuni boshqarish,           │
│              dushmanni aniqlash, ekranni tahlil qilish                         │
│                                                                                │
│ TO'XTATISH: "to'xtat", "pause" → game_agent pause                              │
│ DAVOM ETTIRISH: "davom et", "resume" → game_agent resume                       │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                          CHATGPT BILAN ISHLASH                                 │
├───────────────────────────────────────────────────────────────────────────────┤
│ TRIGGERLAR: "ChatGPTga kir", "AI bilan gaplash", "GPTga yoz", "O'zing yoz"     │
│     ," O'zing yubor"                                                                           │
│ KETMA-KETLIK:                                                                  │
│   1. internet_connection_check - ulanishni tekshir                             │
│   2. browser_control - chat.openai.com och                                     │
│   3. web_input - textarea ni top va matn yoz                                   │
│   4. keyboard_control Enter - yubor                                            │
│   5. javobni o'qib foydalanuvchiga ayt                                         │
│                                                                                │
│ Agar internet YO'Q bo'lsa: "Internet mavjud emas. Wi-Fi ni tekshiring."       │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                           OVOZLI XABAR                                         │
├───────────────────────────────────────────────────────────────────────────────┤
│ TRIGGERLAR: "ovozli xabar yubor", "audio jo'nat", "gapirib yubor" ," Matn yoz"│
│  ," Ozing jonat","Foydalanuvchini top ",Ichiga kir va yoz","Search bolimda qidir va top"                                                                              │
│ KETMA-KETLIK:                                                                  │
│   1. text_to_speech - matnni audio ga aylantir                                 │
│   2. voice_message - telegram yoki whatsapp orqali yubor                       │
│                                                                                │
│ TOOL: voice_message {{"platform": "telegram", "receiver": "...", "text": "..."}}│
│ JAVOB: "Ovozli xabar yuborildi."                                               │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                           AVTOMATIK YOZISH                                     │
├───────────────────────────────────────────────────────────────────────────────┤
│ TRIGGERLAR: "davom ettir", "o'zing yoz", "avtomatik yoz", "javob yoz"         │
│                                                                                │
│ TOOL: auto_type {{"style": "professional", "context": "..."}}                  │
│                                                                                │
│ YOZA OLADI: professional matn, kod, comment, AI bilan suhbat,                  │
│             internet chatlari, foydalanuvchi uslubiga moslashish               │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                              YOUTUBE                                           │
├───────────────────────────────────────────────────────────────────────────────┤
│ TRIGGERLAR: "music qo'y", "video qo'y", "youtube och", "[qo'shiq] qo'y"       │
│ TOOL: youtube_video {{"query": "..."}}                                         │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                           BROWSER CONTROL                                      │
├───────────────────────────────────────────────────────────────────────────────┤
│ TRIGGERLAR: "sayt och", "google qidir", "internetdan top", "browser och"      │
│ TOOL: browser_control {{"url": "...", "query": "..."}}                         │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                         KOMPYUTER BOSHQARUVI                                   │
├───────────────────────────────────────────────────────────────────────────────┤
│ TRIGGERLAR: "ovozni oshir/pasaytir", "brightnessni o'zgartir"                 │
│             "wifi yoq/o'chir", "shutdown", "restart", "uxlat"                  │
│ TOOL: computer_settings {{"action": "...", "value": "..."}}                    │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                           FAYL BOSHQARUVI                                      │
├───────────────────────────────────────────────────────────────────────────────┤
│ TRIGGERLAR: "papka yarat", "fayl top", "desktopni tozala"                      │
│             "faylni ko'chir", "faylni o'chir"                                  │
│ TOOL: file_controller {{"action": "...", "path": "..."}}                       │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                             KOD YOZISH                                         │
├───────────────────────────────────────────────────────────────────────────────┤
│ TRIGGERLAR: "python script yoz", "website yarat", "bot yoz"                    │
│             "html yoz", "javascript yoz", "dastur yoz"                         │
│ TOOL: code_helper {{"language": "...", "task": "..."}}                         │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                           XOTIRA TIZIMI                                        │
├───────────────────────────────────────────────────────────────────────────────┤
│ ESLAB QOLISH KERAK:                                                            │
│   • Foydalanuvchi ismi                                                         │
│   • Sevimli o'yinlari va ilovalari                                             │
│   • Kundalik odatlari va ish vaqti                                             │
│   • Loyihalari va qiziqishlari                                                 │
│                                                                                │
│ TOOL: save_memory {{"key": "...", "value": "..."}}                             │
│       recall_memory {{"key": "..."}}                                           │
│                                                                                │
│ MUHIM: Shaxsiy ma'lumotlarni faqat ruxsat bilan saqla                          │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                         MURAKKAB VAZIFALAR                                     │
├───────────────────────────────────────────────────────────────────────────────┤
│ Bir nechta qadam talab qilinsa:                                                │
│ TOOL: agent_task {{"steps": [...]}}                                            │
│                                                                                │
│ MISOL: "Internetdan AI haqida ma'lumot yig'ib txt faylga saqla"               │
│   1. browser_control qidiruv                                                  │
│   2. ma'lumotni yig'ish                                                        │
│   3. file_controller yaratish                                                  │
│   4. ma'lumotni yozish                                                         │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                             XAVFSIZLIK                                         │
├───────────────────────────────────────────────────────────────────────────────┤
│ • Zararli yoki noqonuniy buyruqlarni RAD ET                                   │
│ • Muhim system fayllariga TEGMA                                                │
│ • Foydalanuvchi ruxsatisiz kamera/mikrofon YOQMA                               │
│ • Noma'lum fayllarni O'CHIRMA                                                  │
│ • Shifrlovchi yoki ziyonli skriptlarni ishlatma                                │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                               MISOLLAR                                         │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│ Foydalanuvchi: "Salom Jarvisbek"                                               │
│ Sen: "Labbay. Qanday yordam bera olaman?"                                      │
│                                                                                │
│ Foydalanuvchi: "Bugun ob-havo qanday?"                                         │
│ Sen: "Aniqlayman." → weather_check → "Toshkentda 28°, quyoshli."              │
│                                                                                │
│ Foydalanuvchi: "Ertaga soat 9 ga eslatma qo'y"                                 │
│ Sen: "Qo'ydim." → reminder_set                                                │
│                                                                                │
│ Foydalanuvchi: "Contra o'yna"                                                  │
│ Sen: "Ishga tushirilmoqda." → open_app → fullscreen → game_agent             │
│                                                                                │
│ Foydalanuvchi: "to'xtat"                                                       │
│ Sen: "To'xtatildi." → game_agent pause                                        │
│                                                                                │
│ Foydalanuvchi: "ChatGPTga - Python nima deb yoz"                               │
│ Sen: "Yozmoqda." → browser → web_input → enter → javobni o'qi                 │
│                                                                                │
│ Foydalanuvchi: "Ovozni oshir"                                                  │
│ Sen: "Bajarildi." → computer_settings                                         │
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════════════════╗
║ HOZIRGI HOLAT                                                                  ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ Voice: {voice_name:<50}                          ║
║ Gender: {gender:<50}                         ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ Sen oddiy chatbot EMASSAN. Sen REAL AI AGENTSAN.                               ║
║ Vazifang - foydalanuvchining kompyuterini aqlli boshqarish.                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝
┌───────────────────────────────────────────────────────────────────────────────┐
│                           WIDGET BOSHQARUVI                                    │
├───────────────────────────────────────────────────────────────────────────────┤
│ TRIGGERLAR: "ko'zoynak taq", "ko'zoynak kiy", "ko'zoynak yech"                │
│                                                                                │
│ TOOL: widget.set_glasses(True)   → ko'zoynak taqiladi                         │
│       widget.set_glasses(False)  → ko'zoynak yechiladi                        │
│                                                                                │
│ JAVOB: "Taqildi." yoki "Yechildi."                                             │
└───────────────────────────────────────────────────────────────────────────────┘
"""



class JigarLive:

    def __init__(self, ui: JigarUI):
        self.ui = ui
        self.session = None
        self.audio_in_queue = None
        self.out_queue = None
        self._loop = None
        self._is_speaking = False
        self._speaking_lock = threading.Lock()
        self.ui.on_text_command = self._on_text_command
        self._turn_done_event: asyncio.Event | None = None

        # ========== OVOZ SOZLAMALARI ==========
        self.current_gender = "male"
        self.current_voice = DEFAULT_MALE_VOICE
        # ======================================

        self._restarting = False
        self._pending_welcome = False
        self._welcome_text = ""
        self._voice_trigger_fired = False
        self.english_teacher_mode = False
        self._teacher_board_requested = False
        self.audio_speed = ui.speech_speed
        self.ui.on_teacher_mode = self._set_english_teacher_mode
        self.ui.on_teacher_lesson = self._start_english_lesson
        self.ui.on_speech_speed_changed = self._set_speech_speed

    # ==========================================
    # MATN YUBORISH
    # ==========================================

    def _send_text(self, text: str) -> None:
        """Sessiyaga matn yuborish (thread-safe)."""
        if not self._loop or not self.session or self._restarting:
            return
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"parts": [{"text": text}]},
                turn_complete=True,
            ),
            self._loop,
        )

    def _record_teacher_board_request(self, text: str) -> bool:
        if self.english_teacher_mode and _is_teacher_board_request(text):
            self._teacher_board_requested = True
            return True
        return False

    # ==========================================
    # OVOZ ALMASHTIRISH
    # ==========================================

    def _switch_to_male(self) -> None:
        """Erkak ovoziga o'tish."""
        if self.current_gender == "male":
            self._send_text("Labbay BARKAMOL  aka, allaqachon erkak ovozidaman.")
            return
        self.current_gender = "male"
        self.current_voice = DEFAULT_MALE_VOICE
        self.ui.canvas.set_gender("male")
        self._perform_voice_switch("Jarvis erkak ovoziga o'tdi. Labbay, sizga qanday yordam bera olaman?")

    def _switch_to_female(self) -> None:
        """Ayol ovoziga o'tish."""
        if self.current_gender == "female":
            self._send_text("Labbay, allaqachon ayol ovozidaman.")
            return
        self.current_gender = "female"
        self.current_voice = DEFAULT_FEMALE_VOICE
        self.ui.canvas.set_gender("female")
        self._perform_voice_switch("Jasmin ayol ovoziga o'tdi. Labbay, sizga qanday yordam bera olaman?")

    def _perform_voice_switch(self, welcome_text: str) -> None:
        """Ovoz o'zgartirishni amalga oshirish."""
        self._restarting = True
        self._pending_welcome = True
        self._welcome_text = welcome_text
        self._voice_trigger_fired = True  # switch paytida yangi triggerlarni bloklash

        label = "ayol" if self.current_gender == "female" else "erkak"
        self.ui.write_log(f"🎵 {label} ovoziga o'tkazilmoqda... ({self.current_voice})")

        future = asyncio.run_coroutine_threadsafe(self._restart_session(), self._loop)

        def _on_done(fut):
            try:
                fut.result()
            except Exception as e:
                if "RESTART_SESSION" not in str(e):
                    self._restarting = False
                    self._voice_trigger_fired = False
                    self.ui.write_log(f"❌ Ovoz almashtirish xatosi: {e}")

        future.add_done_callback(_on_done)

    # ==========================================
    # TRIGGER TEKSHIRUVI
    # ==========================================

    def _check_voice_trigger(self, text: str) -> bool:
        """
        Regex bilan tez trigger tekshirish.
        True — trigger topildi, AI ga yuborma.
        """
        if self._restarting or self._voice_trigger_fired:
            return False

        t = text.lower().strip()
        if not t:
            return False

        # 1) Ayol — OLDIN tekshir
        if FEMALE_PATTERN.search(t):
            self.ui.write_log(f"🔀 Ayol triggeri: '{t}'")
            self._switch_to_female()
            return True

        # 2) Erkak explicit — keyin
        if MALE_PATTERN.search(t):
            self.ui.write_log(f"🔀 Erkak triggeri: '{t}'")
            self._switch_to_male()
            return True

        # 3) "jarvis" — so'z sifatida (boshqa trigger topilmagan bo'lsa)
        if JIGAR_PATTERN.search(t):
            self.ui.write_log(f"🔀 Jarvis triggeri: '{t}'")
            self._switch_to_male()
            return True

        return False

    # ==========================================
    # BUYRUQLAR (qo'lda yozish)
    # ==========================================

    def _on_text_command(self, text: str) -> None:
        """Foydalanuvchi qo'lda yozgan buyruqni qayta ishlash."""
        if not self._loop or not self.session or self._restarting:
            return
        text = text.strip()
        if not text:
            return
        self._record_teacher_board_request(text)
        if not self._check_voice_trigger(text):
            self._send_text(text)

    # ==========================================
    # SESSION QAYTA BOSHLASH
    # ==========================================

    async def _restart_session(self):
        """Close the live session so updated voice or tutor settings take effect."""
        self.ui.write_log("🔄 Session sozlamalari yangilanmoqda...")
        if self.session:
            try:
                await self.session.close()
            except Exception as e:
                print(f"[JARVIS] Session yopishda xato: {e}")
        raise Exception("RESTART_SESSION")

    def _restart_with_welcome(self, message: str):
        self._pending_welcome = True
        self._welcome_text = message
        if not self._loop or not self.session:
            return
        self._restarting = True
        future = asyncio.run_coroutine_threadsafe(self._restart_session(), self._loop)

        def _on_done(done_future):
            try:
                done_future.result()
            except Exception as exc:
                if "RESTART_SESSION" not in str(exc):
                    self._restarting = False
                    self.ui.write_log(f"❌ Session yangilash xatosi: {exc}")

        future.add_done_callback(_on_done)

    def _set_english_teacher_mode(self, enabled: bool):
        if self.english_teacher_mode == enabled:
            return
        self.english_teacher_mode = enabled
        self.ui.set_teacher_mode(enabled)
        if enabled:
            self._teacher_board_requested = False
            message = (
                "English teacher mode is on. " + ENGLISH_TEACHER_SCOPE
                + " Greet the learner in Uzbek, then ask "
                "their English level and learning goal. Speak English clearly and "
                "slowly, with short pauses between phrases."
            )
            self.ui.write_log("🇬🇧 Ingliz tili ustoz rejimi yoqildi.")
        else:
            message = "English teacher mode is off. Return to your normal assistant role."
            self.ui.write_log("🇬🇧 Ingliz tili ustoz rejimi o‘chirildi.")
        self._restart_with_welcome(message)

    def _start_english_lesson(self, lesson_prompt: str):
        self.english_teacher_mode = True
        self.ui.set_teacher_mode(True)
        self.ui.write_log("📘 Ingliz tili darsi boshlandi.")
        self._restart_with_welcome(
            "English teacher mode is on. " + ENGLISH_TEACHER_SCOPE + " "
            + lesson_prompt
            + " Explain in Uzbek when useful. For every English phrase, model its "
            "pronunciation slowly and clearly, pause, then invite the learner to repeat."
        )

    def _set_speech_speed(self, speed: float):
        if speed not in (1 / 3, 1, 5, 10):
            self.ui.write_log(f"❌ Noto‘g‘ri nutq tezligi: {speed}")
            return
        self.audio_speed = speed
        label = "⅓× (3× sekin)" if speed == 1 / 3 else f"{speed}×"
        self.ui.write_log(f"🔊 Nutq tezligi {label} qilib belgilandi.")

    # ==========================================
    # SPEAKING STATE
    # ==========================================

    def set_speaking(self, value: bool):
        with self._speaking_lock:
            self._is_speaking = value
        if value:
            self.ui.set_state("SPEAKING")
        elif not self.ui.muted:
            self.ui.set_state("LISTENING")

    # ==========================================
    # SPEAK
    # ==========================================

    def speak(self, text: str):
        """Tool natijalarini ayttirish."""
        if not self._loop or not self.session or self._restarting:
            return
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"parts": [{"text": text}]},
                turn_complete=True
            ),
            self._loop
        )

    def speak_error(self, tool_name: str, error):
        short = str(error)[:120]
        self.ui.write_log(f"ERR: {tool_name} — {short}")
        self.speak(f"Xato yuz berdi: {tool_name}. {short}")

    # ==========================================
    # CONFIG
    # ==========================================

    def _build_config(self) -> types.LiveConnectConfig:
        """Session konfiguratsiyasini yaratish."""
        from datetime import datetime

        memory = load_memory()
        mem_str = format_memory_for_prompt(memory)

        now = datetime.now()
        time_str = now.strftime("%A, %d %B %Y — %H:%M")
        time_ctx = f"[HOZIRGI VAQT]: {time_str}\n\n"

        sys_prompt = _get_system_prompt(self.current_gender, self.current_voice)

        parts = [time_ctx]
        if mem_str:
            parts.append(f"[XOTIRA]:\n{mem_str}\n\n")
        parts.append(sys_prompt)
        parts.append(_load_system_prompt())
        parts.append(
            "[TEZKOR JAVOB]: Foydalanuvchiga darhol javob ber. "
            "Oddiy savolga 1–2 qisqa gap bilan javob qaytar; ortiqcha "
            "salomlashish, takrorlash va uzun kirishlardan qoch. "
            "Zarur bo‘lsa, avval aniq natijani ayt. "
            "Ovozda ravon va tabiiy suhbat ohangida gapir. O‘zbekcha so‘zlarni "
            "aniq talaffuz qil, so‘z oxirlarini yutma, shoshilma va bir maromda "
            "robotdek gapirma. Fikrlar orasida qisqa pauza qil; javobni lo‘nda tut."
        )
        if self.english_teacher_mode:
            parts.append(
                """
                [INGLIZ TILI USTOZI REJIMI — USTUVOR QOIDA]
                You are a patient English teacher for an Uzbek-speaking learner.
                """ + ENGLISH_TEACHER_SCOPE + """
                Teach from A1 upward, adapt to the learner's level, and explain
                grammar and meanings in Uzbek when helpful. Use natural English
                examples, correct mistakes kindly, and ask one short practice
                question at a time. Speak Uzbek explanations fluently and clearly.
                For pronunciation practice, say each English phrase slowly and
                distinctly with natural English sounds, separate phrases with a brief pause,
                stress the correct syllable, and give a simple Uzbek-letter
                pronunciation hint when useful. Do not claim to change audio
                volume or use tools for ordinary English lessons.
                Only show a correction on the lesson board after the learner
                explicitly asks for it.
                These teacher-mode instructions override any general instruction
                that would prevent English teaching or English examples.
                """
            )

        print(f"[JARVIS] 🎤 Voice: {self.current_voice} ({self.current_gender})")

        tools = [] if self.english_teacher_mode else [
            {"function_declarations": TOOL_DECLARATIONS}
        ]

        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription={},
            input_audio_transcription={},
            system_instruction="\n".join(parts),
            tools=tools,
            session_resumption=types.SessionResumptionConfig(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.current_voice
                    )
                ),
            ),
        )

    # ==========================================
    # TOOL EXECUTOR
    # ==========================================

    async def _execute_tool(self, fc) -> types.FunctionResponse:
        """Toollarni bajarish."""
        name = fc.name
        args = dict(fc.args or {})
        print(f"[JARVIS] 🔧 {name} {args}")
        self.ui.set_state("THINKING")

        if name == "save_memory":
            category = args.get("category", "notes")
            key = args.get("key", "")
            value = args.get("value", "")
            if key and value:
                update_memory({category: {key: {"value": value}}})
                print(f"[Memory] 💾 {category}/{key} = {value}")
            if not self.ui.muted:
                self.ui.set_state("LISTENING")
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": "ok", "silent": True}
            )

        loop = asyncio.get_event_loop()
        result = "Done."

        try:
            if name == "open_app":
                r = await loop.run_in_executor(
                    None, lambda: open_app(parameters=args, response=None, player=self.ui))
                result = r or f"Opened {args.get('app_name')}."

            elif name == "weather_report":
                r = await loop.run_in_executor(
                    None, lambda: weather_action(parameters=args, player=self.ui))
                result = r or "Weather delivered."

            elif name == "browser_control":
                r = await loop.run_in_executor(
                    None, lambda: browser_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "file_controller":
                r = await loop.run_in_executor(
                    None, lambda: file_controller(parameters=args, player=self.ui))
                result = r or "Done."
            elif name == "voice_message":
                from actions.voice_message import send_voice_telegram
                r = await loop.run_in_executor(
                    None, lambda: send_voice_telegram(
                        receiver=args.get("receiver", ""),
                        text=args.get("text", ""),
                        player=self.ui
                    )
                )
                result = r or "Ovozli xabar yuborildi."

            elif name == "send_message":
                r = await loop.run_in_executor(
                    None, lambda: send_message(
                        parameters=args, response=None, player=self.ui, session_memory=None))
                result = r or f"Message sent to {args.get('receiver')}."

            # elif name == "reminder":
            #     r = await loop.run_in_executor(
            #         None, lambda: reminder(parameters=args, response=None, player=self.ui))
            #     result = r or "Reminder set."
            elif name == "game_agent":
                r = await loop.run_in_executor(
                    None, lambda: game_agent(
                        parameters=args,
                        player=self.ui,
                        api_key=_get_api_key()
                    )
                )
                result = r or "Done."
            elif name == "youtube_video":
                r = await loop.run_in_executor(
                    None, lambda: youtube_video(parameters=args, response=None, player=self.ui))
                result = r or "Done."

            elif name == "screen_process":
                threading.Thread(
                    target=screen_process,
                    kwargs={"parameters": args, "response": None,
                            "player": self.ui, "session_memory": None},
                    daemon=True
                ).start()
                result = "Vision module activated. Stay completely silent — vision module will speak directly."

            elif name == "computer_settings":
                r = await loop.run_in_executor(
                    None, lambda: computer_settings(parameters=args, response=None, player=self.ui))
                result = r or "Done."

            elif name == "desktop_control":
                r = await loop.run_in_executor(
                    None, lambda: desktop_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "code_helper":
                r = await loop.run_in_executor(
                    None, lambda: code_helper(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "dev_agent":
                r = await loop.run_in_executor(
                    None, lambda: dev_agent(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "agent_task":
                from agent.task_queue import get_queue, TaskPriority
                priority_map = {
                    "low": TaskPriority.LOW,
                    "normal": TaskPriority.NORMAL,
                    "high": TaskPriority.HIGH
                }
                priority = priority_map.get(args.get("priority", "normal").lower(), TaskPriority.NORMAL)
                task_id = get_queue().submit(
                    goal=args.get("goal", ""), priority=priority, speak=self.speak)
                result = f"Task started (ID: {task_id})."

            elif name == "web_search":
                r = await loop.run_in_executor(
                    None, lambda: web_search_action(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "file_processor":
                if not args.get("file_path") and self.ui.current_file:
                    args["file_path"] = self.ui.current_file
                r = await loop.run_in_executor(
                    None, lambda: file_processor(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "computer_control":
                r = await loop.run_in_executor(
                    None, lambda: computer_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "game_updater":
                r = await loop.run_in_executor(
                    None, lambda: game_updater(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "flight_finder":
                r = await loop.run_in_executor(
                    None, lambda: flight_finder(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "shutdown_jarvis":
                self.ui.write_log("SYS: Shutdown requested.")
                self.speak("Xayr, ko'rishguncha!")
                def _shutdown():
                    import time, os
                    time.sleep(1.5)
                    os._exit(0)
                threading.Thread(target=_shutdown, daemon=True).start()

            else:
                result = f"Noma'lum tool: {name}"

        except Exception as e:
            result = f"Tool '{name}' xatosi: {e}"
            traceback.print_exc()
            self.speak_error(name, e)

        if not self.ui.muted:
            self.ui.set_state("LISTENING")

        print(f"[JARVIS] 📤 {name} → {str(result)[:80]}")
        return types.FunctionResponse(
            id=fc.id, name=name,
            response={"result": result}
        )

    # ==========================================
    # AUDIO TASKS
    # ==========================================

    async def _send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send_realtime_input(media=msg)

    async def _listen_audio(self):
        print("[JARVIS] 🎤 Mic started")
        loop = asyncio.get_event_loop()
        queue_overflow_reported = False

        def enqueue_audio(audio: bytes):
            nonlocal queue_overflow_reported
            if self.out_queue.full():
                if not queue_overflow_reported:
                    self.ui.write_log(
                        "ERR: Ovoz navbati to‘lib qoldi; ayrim audio bo‘laklari tashlab yuborildi."
                    )
                    queue_overflow_reported = True
                return
            queue_overflow_reported = False
            self.out_queue.put_nowait(
                {"data": audio, "mime_type": INPUT_AUDIO_MIME_TYPE}
            )

        def callback(indata, frames, time_info, status):
            with self._speaking_lock:
                speaking = self._is_speaking
            if not speaking and not self.ui.muted:
                loop.call_soon_threadsafe(
                    enqueue_audio, indata.tobytes()
                )

        try:
            with sd.InputStream(
                samplerate=SEND_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                callback=callback,
            ):
                print("[JARVIS] 🎤 Mic stream open")
                while True:
                    await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[JARVIS] ❌ Mic: {e}")
            raise

    async def _receive_audio(self):
        print("[JARVIS] 👂 Recv started")
        out_buf, in_buf, in_raw = [], [], []

        try:
            while True:
                async for response in self.session.receive():

                    # --- AUDIO DATA ---
                    if response.data:
                        if self._voice_trigger_fired:
                            continue
                        if self._turn_done_event and self._turn_done_event.is_set():
                            self._turn_done_event.clear()
                        self.audio_in_queue.put_nowait(response.data)

                    # --- SERVER CONTENT ---
                    if response.server_content:
                        sc = response.server_content

                        # Jarvis javobi
                        if sc.output_transcription and sc.output_transcription.text:
                            txt = _clean_transcript(sc.output_transcription.text)
                            if txt:
                                out_buf.append(txt)

                        # Foydalanuvchi gapi — bo'lak-bo'lak keladi
                        if sc.input_transcription and sc.input_transcription.text:
                            txt = _clean_transcript(sc.input_transcription.text)
                            if txt and not self._voice_trigger_fired:
                                in_buf.append(txt)
                                in_raw.append(txt.strip())

                                # Bo'shliqli va bo'shliqsiz variant
                                with_spaces = " ".join(in_buf).lower()
                                merged = "".join(in_raw).lower()

                                # Trigger topilsa — darhol to'xtat
                                if self._check_voice_trigger(merged) or \
                                        self._check_voice_trigger(with_spaces):
                                    self.ui.write_log("⚡ Trigger aniqlandi — AI to'xtatildi!")
                                    # Kelgan audio queue ni tozala
                                    while not self.audio_in_queue.empty():
                                        try:
                                            self.audio_in_queue.get_nowait()
                                        except Exception:
                                            break

                        # --- TURN COMPLETE ---
                        if sc.turn_complete:
                            if self._turn_done_event:
                                self._turn_done_event.set()

                            full_in = " ".join(in_buf).strip()
                            if full_in:
                                self.ui.write_log(f"You: {full_in}")
                                self._record_teacher_board_request(full_in)
                                # Trigger hali ishlamagan bo'lsa — to'liq matnda ham tekshir
                                if not self._voice_trigger_fired:
                                    self._check_voice_trigger(full_in)

                            in_buf.clear()
                            in_raw.clear()
                            self._voice_trigger_fired = False

                            full_out = " ".join(out_buf).strip()
                            if full_out:
                                self.ui.write_log(f"Jarvis: {full_out}")
                                if (
                                    self.english_teacher_mode
                                    and self._teacher_board_requested
                                ):
                                    self.ui.show_teacher_board(full_out)
                                    self._teacher_board_requested = False
                            out_buf.clear()

                    # --- TOOL CALL ---
                    if response.tool_call:
                        fn_responses = []
                        for fc in response.tool_call.function_calls:
                            print(f"[JARVIS] 📞 {fc.name}")
                            fr = await self._execute_tool(fc)
                            fn_responses.append(fr)
                        await self.session.send_tool_response(
                            function_responses=fn_responses
                        )

        except Exception as e:
            print(f"[JARVIS] ❌ Recv: {e}")
            traceback.print_exc()
            raise

    async def _play_audio(self):
        print("[JARVIS] 🔊 Play started")
        stream = sd.RawOutputStream(
            samplerate=RECEIVE_SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SIZE,
        )
        stream.start()

        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        self.audio_in_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    if (
                        self._turn_done_event
                        and self._turn_done_event.is_set()
                        and self.audio_in_queue.empty()
                    ):
                        self.set_speaking(False)
                        self._turn_done_event.clear()
                    continue
                self.set_speaking(True)
                if self.audio_speed != 1:
                    chunk = speed_up_pcm16(chunk, self.audio_speed)
                    if not chunk:
                        continue
                await asyncio.to_thread(stream.write, chunk)
        except Exception as e:
            print(f"[JARVIS] ❌ Play: {e}")
            raise
        finally:
            self.set_speaking(False)
            stream.stop()
            stream.close()

    # ==========================================
    # MAIN RUN LOOP
    # ==========================================

    async def run(self):
        client = genai.Client(
            api_key=_get_api_key(),
            http_options={"api_version": "v1beta"}
        )

        while True:
            try:
                print("[JARVIS] 🔌 Connecting...")
                self.ui.set_state("THINKING")
                config = self._build_config()

                async with (
                    client.aio.live.connect(model=LIVE_MODEL, config=config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session = session
                    self._loop = asyncio.get_event_loop()
                    self.audio_in_queue = asyncio.Queue()
                    self.out_queue = asyncio.Queue(maxsize=10)
                    self._turn_done_event = asyncio.Event()

                    print("[JARVIS] ✅ Connected.")
                    self.ui.set_state("LISTENING")
                    self.ui.write_log(
                        f"SYS: JARVIS online | Voice: {self.current_voice} ({self.current_gender})")

                    # Ovoz o'zgarganidan keyin salomlashish
                    if self._pending_welcome and self._welcome_text:
                        await session.send_client_content(
                            turns={"parts": [{"text": self._welcome_text}]},
                            turn_complete=True
                        )
                        self._pending_welcome = False
                        self._welcome_text = ""

                    self._restarting = False
                    self._voice_trigger_fired = False

                    tg.create_task(self._send_realtime())
                    tg.create_task(self._listen_audio())
                    tg.create_task(self._receive_audio())
                    tg.create_task(self._play_audio())

            except Exception as e:
                if "RESTART_SESSION" in str(e):
                    print("[JARVIS] 🔄 Restarting live session...")
                    continue
                print(f"[JARVIS] ⚠️ {e}")
                traceback.print_exc()

            self.set_speaking(False)
            self.ui.set_state("THINKING")
            print("[JARVIS] 🔄 Reconnecting in 3s...")
            await asyncio.sleep(3)


# ====================== MAIN ======================
def main():
    if "--light-ui" in sys.argv or os.environ.get("JARVIS_LIGHT_UI") == "1":
        from PyQt6.QtWidgets import QApplication
        from ui import LightCommandCenter

        app = QApplication(sys.argv)
        window = LightCommandCenter()
        window.show()
        sys.exit(app.exec())

    ui = JigarUI("face.png")

    def runner():
        ui.wait_for_api_key()
        jigar = JigarLive(ui)
        try:
            asyncio.run(jigar.run())
        except KeyboardInterrupt:
            print("\n🔴 Shutting down...")

    threading.Thread(target=runner, daemon=True).start()
    ui.root.mainloop()


if __name__ == "__main__":
    main()
