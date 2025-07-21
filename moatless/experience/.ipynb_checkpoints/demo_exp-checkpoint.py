import json
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


original_pool = {
    "search": [
        {
            "tool": "FindClass",
            "task": "I need to see the implementation of the DatabaseManager class to understand how it handles transactions",
            "file_pattern": None,
            "class_name": "DatabaseManager"
        },
        {
            "tool": "FindClass",
            "task": "Show me the UserAuthentication class in the auth module",
            "file_pattern": "auth/*.py",
            "class_name": "UserAuthentication"
        },
        {
            "tool": "FindFunction",
            "task": "Find the calculate_interest function in our financial module to review its logic",
            "file_pattern": "financial/**/*.py",
            "function_name": "calculate_interest",
            "class_name": None
        },
        {
            "tool": "FindFunction",
            "task": "Show me the validate_token method in the JWTAuthenticator class",
            "file_pattern": None,
            "function_name": "validate_token",
            "class_name": "JWTAuthenticator"
        },
        {
            "tool": "FindCodeSnippet",
            "task": "I need to understand how the User class is structured in our authentication system. Let me find its definition.",
            "file_pattern": None,
            "code_snippet": "class User(BaseModel):"
        },
        {
            "tool": "FindCodeSnippet",
            "task": "The system seems to use a default timeout value. I should check where DEFAULT_TIMEOUT is defined in the configuration.",
            "file_pattern": "**/config/*.py",
            "code_snippet": "DEFAULT_TIMEOUT ="
        },
        {
            "tool": "FindCodeSnippet",
            "task": "To understand how request processing works, I need to examine the _handlers dictionary in the processor service.",
            "file_pattern": "services/processor.py",
            "code_snippet": "_handlers ="
        },
        {
            "tool": "SemanticSearch",
            "task": "Find all implementations of database connection pooling in our codebase",
            "file_pattern": None,
            "query": "database connection pooling implementation",
            "category": "implementation"
        },
        {
            "tool": "SemanticSearch",
            "task": "We need to find all test cases related to user authentication in our test suite",
            "file_pattern": "tests/*.py",
            "query": "user authentication test cases",
            "category": "test"
        }
    ],
    "view": [
        {
            "tool": "ViewCode",
            "task": "Show me the implementation of the authenticate method in the AuthenticationService class",
            "files": [{"file_path": "auth/service.py", "start_line": None, "end_line": None, "span_ids": ["AuthenticationService.authenticate"]}]
        },
        {
            "tool": "ViewCode",
            "task": "Show me lines 50-75 of the database configuration file",
            "files": [{"file_path": "config/database.py", "start_line": 50, "end_line": 75, "span_ids": []}]
        }
    ],
    "modify": [
        {
            "tool": "StringReplace",
            "task": "Update the error message in the validate_user method",
            "path": "auth/validator.py",
            "old_str": "    if not user.is_active:\n        raise ValueError(\"Invalid user\")\n    return user",
            "new_str": "    if not user.is_active:\n        raise ValueError(f\"Invalid user: {username} does not meet the required criteria\")\n    return user"
        },
        {
            "tool": "StringReplace",
            "task": "Update the user validation logic",
            "path": "auth/validator.py",
            "old_str": "def validate_user(username, password):\n    if len(username) < 3:\n        return False\n    if len(password) < 8:\n        return False\n    return True",
            "new_str": "def validate_user(username, password):\n    if len(username) < 3 or not is_valid_email(username):\n        return False\n    if len(password) < 12 or not has_special_chars(password):\n        return False\n    if not has_numbers(password):\n        return False\n    return True"
        },
        {
            "tool": "StringReplace",
            "task": "Add a new helper function to validate passwords",
            "path": "auth/validator.py",
            "old_str": "def validate_user(username, password):\n    if len(username) < 3 or not is_valid_email(username):\n        return False\n    if len(password) < 12 or not has_special_chars(password):\n        return False\n    if not has_numbers(password):\n        return False\n    return True",
            "new_str": "def is_password_complex(password: str) -> bool:\n    has_uppercase = any(c.isupper() for c in password)\n    has_lowercase = any(c.islower() for c in password)\n    has_digit = any(c.isdigit() for c in password)\n    has_special = any(not c.isalnum() for c in password)\n    return all([has_uppercase, has_lowercase, has_digit, has_special])\n\ndef validate_user(username, password):\n    if len(username) < 3 or not is_valid_email(username):\n        return False\n    if len(password) < 12 or not has_special_chars(password):\n        return False\n    if not has_numbers(password) or not is_password_complex(password):\n        return False\n    return True"
        },
        {
            "tool": "StringReplace",
            "task": "Remove the deprecated logging configuration",
            "path": "utils/logger.py",
            "old_str": "    # Legacy logging configuration\n    if legacy_mode:\n        logging.basicConfig(\n            level=logging.DEBUG,\n            format=\"[%(levelname)s] %(message)s\",\n            filename=\"legacy.log\"\n        )\n        logger.warning(\"Using deprecated legacy logging configuration\")",
            "new_str": ""
        },
        {
            "tool": "StringReplace",
            "task": "Add a new test case for password validation with special characters",
            "path": "tests/test_validator.py",
            "old_str": "def test_validate_user():\n    # Test basic validation\n    assert validate_user(\"bob@example.com\", \"password123\") is True\n    assert validate_user(\"alice@example.com\", \"short\") is False\n    \n    # Test email format\n    assert validate_user(\"invalid-email\", \"password123\") is False",
            "new_str": "def test_validate_user():\n    # Test basic validation\n    assert validate_user(\"bob@example.com\", \"password123\") is True\n    assert validate_user(\"alice@example.com\", \"short\") is False\n    \n    # Test email format\n    assert validate_user(\"invalid-email\", \"password123\") is False\n\ndef test_validate_password_special_chars():\n    # Test passwords with special characters\n    assert validate_user(\"bob@example.com\", \"Pass!@#123\") is True\n    assert validate_user(\"alice@example.com\", \"NoSpecialChars123\") is False\n    assert validate_user(\"carol@example.com\", \"!@#$%^&*(\") is False  # No alphanumeric chars"
        }
    ],
    "finish": [
        {
            "tool": "Finish",
            "task": "Fix the bug in the date parsing logic",
            "thoughts": "I've fixed the date parsing bug and added tests to prevent regression",
            "finish_reason": "Fixed date parsing bug that was incorrectly handling timezone conversions"
        },
        {
            "tool": "Finish",
            "task": "Add input validation to the process_order function",
            "thoughts": "I've added input validation and comprehensive tests to verify all validation cases",
            "finish_reason": "Added robust input validation to process_order function with proper error handling"
        }
    ]
}


class DemoPool(BaseModel):
    pool: Optional[Dict[str, Any]] = Field(
        ..., description="Demonstrations pool for assistant"
    )
    
    def __init__(self, pool=None):
        if pool is None:
            pool = original_pool
        super().__init__(pool=pool)

    def generate_few_shots(self, ty:str):
        demos = self.pool[ty]
        examples = "\n\n# Examples\nHere are some examples of how to use the available actions:\n\n"
        for demo in demos:
            extracted_dict = {key: demo[key] for key in demo.keys() if key != 'task'}
            json_str = json.dumps(extracted_dict)
            result_str = f'Task: {demo["task"]}\n{json_str}'
            examples += result_str + "\n\n"
        return examples

    def add_shots():
        pass

    def update_shots():
        pass
        