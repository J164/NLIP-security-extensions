"""NLIP message models defined by ECMA-430."""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterator

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


def _same_text(left: str | None, right: str | None) -> bool:
    return left is not None and right is not None and left.casefold() == right.casefold()


class CaseInsensitiveEnum(str, Enum):
    """Accept protocol enum values regardless of JSON string casing."""

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str):
            value = value.casefold()
            return next((member for member in cls if member.value.casefold() == value), None)
        return None


class AllowedFormat(CaseInsensitiveEnum):
    text = "text"
    token = "token"
    structured = "structured"
    binary = "binary"
    location = "location"
    generic = "generic"


class ReservedToken(CaseInsensitiveEnum):
    """Reserved NLIP token prefixes and the control message type."""

    authorization = "authorization"
    conversation = "conversation"
    control = "control"

    @classmethod
    def is_reserved(cls, value: str | None) -> bool:
        return cls.is_authorization(value) or cls.is_conversation(value)

    @classmethod
    def is_authorization(cls, value: str | None) -> bool:
        return value is not None and value.casefold().startswith(cls.authorization.value)

    @classmethod
    def is_conversation(cls, value: str | None) -> bool:
        return value is not None and value.casefold().startswith(cls.conversation.value)

    @classmethod
    def is_control(cls, value: str | None) -> bool:
        return _same_text(value, cls.control.value)

    @classmethod
    def get_suffix(cls, value: str, separator: str = "") -> str:
        for prefix in (cls.authorization.value, cls.conversation.value):
            if value.casefold().startswith(prefix):
                return value[len(prefix) + len(separator) :].strip()
        return value


class _NLIPModel(BaseModel):
    """Strict base model shared by messages and submessages."""

    # Unknown fields are rejected at the protocol boundary; assignment is validated too.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_field_names(cls, value: Any) -> Any:
        """Map case-insensitive wire names such as ``MessageType`` to model fields."""
        if not isinstance(value, dict):
            return value

        aliases = {
            "messagetype": "messagetype",
            "message_type": "messagetype",
            "format": "format",
            "subformat": "subformat",
            "content": "content",
            "label": "label",
            "submessages": "submessages",
        }
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = aliases.get(str(key).casefold(), key)
            if normalized_key in normalized:
                raise ValueError(f"Duplicate NLIP field: {key}")
            normalized[normalized_key] = item
        return normalized


class NLIPSubMessage(_NLIPModel):
    """A secondary content part attached to an :class:`NLIPMessage`."""

    format: AllowedFormat
    subformat: str
    content: JsonValue
    label: str | None = None

    def update_content(self, content: JsonValue) -> None:
        self.content = content

    def matches(
        self,
        format: AllowedFormat | str,
        subformat: str | None = None,
        label: str | None = None,
    ) -> bool:
        return (
            self.format == AllowedFormat(format)
            and (subformat is None or _same_text(self.subformat, subformat))
            and (label is None or _same_text(self.label, label))
        )

    def extract_field(
        self,
        format: AllowedFormat | str,
        subformat: str | None = None,
        label: str | None = None,
    ) -> JsonValue | None:
        return self.content if self.matches(format, subformat, label) else None


class NLIPMessage(_NLIPModel):
    """One primary content part plus optional heterogeneous submessages."""

    messagetype: str | None = None
    format: AllowedFormat
    subformat: str
    content: JsonValue
    submessages: list[NLIPSubMessage] | None = Field(default=None, min_length=1)

    @classmethod
    def text(
        cls,
        content: str,
        language: str = "English",
        messagetype: str | None = None,
    ) -> "NLIPMessage":
        return cls(
            messagetype=messagetype,
            format=AllowedFormat.text,
            subformat=language,
            content=content,
        )

    @classmethod
    def control(cls, content: str, language: str = "English") -> "NLIPMessage":
        return cls.text(content, language, ReservedToken.control.value)

    @classmethod
    def token(cls, content: str, token_type: str) -> "NLIPMessage":
        return cls(format=AllowedFormat.token, subformat=token_type, content=content)

    @classmethod
    def structured(cls, content: JsonValue, subformat: str = "json") -> "NLIPMessage":
        return cls(format=AllowedFormat.structured, subformat=subformat, content=content)

    def is_control(self) -> bool:
        return _same_text(self.messagetype, ReservedToken.control.value)

    def iter_parts(self) -> Iterator[NLIPSubMessage]:
        """Present the primary message and submessages through one read-only view."""
        # The primary fields have the same shape as a submessage, so extraction code
        # does not need separate branches for primary and secondary content.
        yield NLIPSubMessage(
            format=self.format,
            subformat=self.subformat,
            content=self.content,
        )
        yield from self.submessages or []

    def token_submessages(self) -> list[NLIPSubMessage]:
        return [part for part in self.iter_parts() if part.format == AllowedFormat.token]

    def add_submessage(self, submessage: NLIPSubMessage) -> None:
        if self.submessages is None:
            self.submessages = [submessage]
        else:
            self.submessages.append(submessage)

    def add_token(self, content: str, token_type: str, label: str | None = None) -> None:
        """Attach a token unless an equivalent token is already present."""
        token = NLIPSubMessage(
            format=AllowedFormat.token,
            subformat=token_type,
            content=content,
            label=label,
        )
        if not self.has_token(token):
            self.add_submessage(token)

    def add_conversation_token(
        self,
        content: str,
        force_change: bool = False,
        label: str | None = None,
    ) -> None:
        existing = self.extract_conversation_token(label)
        if existing is None:
            self.add_token(content, ReservedToken.conversation.value, label)
        elif force_change:
            if self.format == AllowedFormat.token and ReservedToken.is_conversation(
                self.subformat
            ):
                self.content = content
            for token in self.submessages or []:
                if ReservedToken.is_conversation(token.subformat) and (
                    label is None or _same_text(token.label, label)
                ):
                    token.content = content

    def add_authentication_token(self, content: str, label: str | None = None) -> None:
        if self.extract_authentication_token(label) is None:
            self.add_token(content, ReservedToken.authorization.value, label)

    def has_token(self, candidate: NLIPSubMessage) -> bool:
        return any(
            _same_text(token.subformat, candidate.subformat)
            and token.content == candidate.content
            and (
                token.label == candidate.label
                or _same_text(token.label, candidate.label)
            )
            for token in self.token_submessages()
        )

    def echo_tokens_from(self, request: "NLIPMessage") -> None:
        """Copy peer-created tokens into a response as required by ECMA-430 §6.2."""
        for token in request.token_submessages():
            if not self.has_token(token):
                self.add_submessage(token.model_copy(deep=True))

    def extract_fields(
        self,
        format: AllowedFormat | str,
        subformat: str | None = None,
        label: str | None = None,
    ) -> list[JsonValue]:
        return [
            part.content
            for part in self.iter_parts()
            if part.matches(format, subformat, label)
        ]

    def extract_field(
        self,
        format: AllowedFormat | str,
        subformat: str | None = None,
        label: str | None = None,
    ) -> JsonValue | None:
        values = self.extract_fields(format, subformat, label)
        return values[0] if values else None

    def find_labeled_submessage(self, label: str) -> NLIPSubMessage | None:
        return next(
            (
                submessage
                for submessage in self.submessages or []
                if _same_text(submessage.label, label)
            ),
            None,
        )

    def extract_text(self, language: str | None = "English", separator: str = " ") -> str | None:
        """Join matching text parts; pass ``language=None`` to accept any language."""
        values = self.extract_fields(AllowedFormat.text, language)
        return separator.join(str(value) for value in values) if values else None

    def extract_token(self, token_type: str, label: str | None = None) -> str | None:
        values = self.extract_fields(AllowedFormat.token, token_type, label)
        return str(values[0]) if values else None

    def extract_conversation_token(self, label: str | None = None) -> str | None:
        for token in self.token_submessages():
            if ReservedToken.is_conversation(token.subformat) and (
                label is None or _same_text(token.label, label)
            ):
                return str(token.content)
        return None

    def extract_authentication_token(self, label: str | None = None) -> str | None:
        for token in self.token_submessages():
            if ReservedToken.is_authorization(token.subformat) and (
                label is None or _same_text(token.label, label)
            ):
                return str(token.content)
        return None

    def to_dict(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json", exclude_none=True)

    def to_json(self) -> str:
        return self.model_dump_json(exclude_none=True)
