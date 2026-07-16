import logging
from typing import Any, Dict, Optional


class SessionStore:
    """
    Generic session-scoped singleton store.

    Each session holds a flat dict of named slots. Any component can read/write
    its own slot by key. Example keys:
        "delivery_status"  — artifact lifecycle tracking
        (future keys as needed)
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._stores: Dict[str, Dict[str, Any]] = {}
        return cls._instance

    def _ensure_session(self, session_id: str) -> None:
        if session_id not in self._stores:
            self._stores[session_id] = {}

    def get(self, session_id: str, key: str, default: Any = None) -> Any:
        return self._stores.get(session_id, {}).get(key, default)

    def set(self, session_id: str, key: str, value: Any) -> None:
        self._ensure_session(session_id)
        self._stores[session_id][key] = value

    def delete(self, session_id: str, key: str) -> None:
        store = self._stores.get(session_id)
        if store:
            store.pop(key, None)

    def get_session(self, session_id: str) -> Dict[str, Any]:
        return dict(self._stores.get(session_id, {}))

    def clear_session(self, session_id: str) -> None:
        self._stores.pop(session_id, None)
        logging.debug("[SessionStore] Cleared session: %s", session_id)


session_store = SessionStore()
