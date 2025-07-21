# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     find_called_objects
   Description :
   Author :       Silin
   date：          2025/3/10
-------------------------------------------------
   Change Activity:
                   2025/3/10:
-------------------------------------------------
"""
import logging
from fnmatch import fnmatch
from typing import List, Optional, Tuple, Type, ClassVar

from pydantic import Field, model_validator

from moatless.actions.model import ActionArguments, FewShotExample
from moatless.actions.search_base import SearchBaseAction, SearchBaseArgs
from moatless.file_context import FileContext

logger = logging.getLogger(__name__)


class FindCalledObjectArgs(SearchBaseArgs):
    """
    这个函数就是FindCodeSnippet套了一层壳，输入是模型认为有用的调用对象的名字，会返回一段代码中被调用的对象的具体实现
    """

    called_object: str = Field(..., description="The exact called object to find.")
    file_pattern: Optional[str] = Field(
        default=None,
        description="A glob pattern to filter search results to specific file types or directories. ",
    )

    class Config:
        title = "FindCalledObject"

    @model_validator(mode="after")
    def validate_snippet(self) -> "FindCalledObjectArgs":
        if not self.called_object.strip():
            raise ValueError("called object cannot be empty")
        return self

    def to_prompt(self):
        prompt = f"Searching for called object: {self.called_object}"
        if self.file_pattern:
            prompt += f" in files matching the pattern: {self.file_pattern}"
        return prompt

    def short_summary(self) -> str:
        param_str = f"called_object={self.called_object}"
        if self.file_pattern:
            param_str += f", file_pattern={self.file_pattern}"
        return f"{self.name}({param_str})"


class FindCalledObject(SearchBaseAction):
    args_schema: ClassVar[Type[ActionArguments]] = FindCalledObjectArgs

    max_hits: int = Field(
        10,
        description="The maximum number of search results to return. Default is 10.",
    )

    def _search_for_context(
        self, args: FindCalledObjectArgs
    ) -> Tuple[FileContext, bool]:
        logger.info(
            f"{self.name}: {args.called_object} (file_pattern: {args.file_pattern})"
        )

        matches = self._repository.find_exact_matches(
            search_text=args.called_object, file_pattern=args.file_pattern
        )

        if args.file_pattern and len(matches) > 1:
            matches = [
                (file_path, line_num)
                for file_path, line_num in matches
                if fnmatch(file_path, args.file_pattern)
            ]

        search_result_context = FileContext(repo=self._repository)
        for file_path, start_line in matches[: self.max_hits]:
            num_lines = len(args.called_object.splitlines())
            end_line = start_line + num_lines - 1

            search_result_context.add_line_span_to_context(
                file_path, start_line, end_line, add_extra=False
            )

        return search_result_context, False

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input='''The user's location is empty, but the location is update by the profile and I need to find the object associated with the user's location that is called in the code but not implemented in the code.''',
                action=FindCalledObjectArgs(
                    thoughts="The user's location is defined via user.update_location(profile.location), profile is called, I need to look further into the 'class Profile'",
                    called_object="class Profile",
                    file_pattern="**/profile.py",
                ),
            ),
            FewShotExample.create(
                user_input="This code seems to use DEFAULT_TIMEOUT variable to initialize the system. However, DEFAULT_TIMEOUT doesn't seem to be defined in the current code, I need to search further for DEFAULT_TIMEOUT.",
                action=FindCalledObjectArgs(
                    thoughts="To find the timeout configuration, I'll search for the exact variable declaration 'DEFAULT_TIMEOUT =' in config files",
                    called_object="DEFAULT_TIMEOUT =",
                    file_pattern="**/config/*.py",
                ),
            ),
            # FewShotExample.create(
            #     user_input="telephone_boos = {'silin': phone1, 'han': phone2}\n\nThis code tries to get the name of silin to map phone1, but the value of phone1 does not yet appear in the current code, and I need to search further for the value of phone1",
            #     action=FindCalledObjectArgs(
            #         thoughts="To find the timeout configuration, I'll search for the exact variable declaration 'DEFAULT_TIMEOUT =' in config files",
            #         called_object="DEFAULT_TIMEOUT =",
            #     ),
            # ),
            FewShotExample.create(
                user_input='''This code uses the handling function to get the result, but the handling function is not in the code I see, I need to search for the implementation code of the handling.''',
                action=FindCalledObjectArgs(
                    thoughts="To find the handling function, I'll search for the exact implementation code of 'def handling'.",
                    called_object="def handling",
                    file_pattern="**/handling.py",
                ),
            ),
        ]

