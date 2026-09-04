"""
Unit tests for per-provider ownership/sharing visibility on MusicController.

These cover the ownership/sharing model that replaced the per-user ``provider_filter`` as
the music-provider visibility mechanism: each music provider instance declares an ``owner``
(a user_id, or empty for a house/unowned provider) and a ``shared`` flag (default True).
"""

from __future__ import annotations

from unittest.mock import Mock

from music_assistant_models.auth import UserRole
from music_assistant_models.enums import ProviderType

from music_assistant.constants import CONF_OWNER, CONF_SHARED
from music_assistant.controllers.music import MusicController

ALICE = "user-alice"
BOB = "user-bob"


def _make_prov(instance_id: str, prov_type: ProviderType = ProviderType.MUSIC) -> Mock:
    prov = Mock()
    prov.instance_id = instance_id
    prov.type = prov_type
    return prov


def _make_user(user_id: str, role: UserRole = UserRole.USER) -> Mock:
    return Mock(user_id=user_id, role=role)


def _controller(provider_config: dict[str, dict[str, object]]) -> MusicController:
    """
    Build a bare MusicController whose config serves the given per-provider values.

    :param provider_config: Mapping of provider instance id -> {owner: ..., shared: ...}. A
        provider missing a key falls back to the passed default, mirroring an install that
        never stored the value (i.e. the default-shared, unowned state).
    """

    def _get_raw(instance_id: str, key: str, default: object = None) -> object:
        return provider_config.get(instance_id, {}).get(key, default)

    controller = MusicController.__new__(MusicController)
    controller.mass = Mock()
    controller.mass.config.get_raw_provider_config_value.side_effect = _get_raw
    return controller


def test_admin_sees_every_provider() -> None:
    """An administrator is never subject to ownership/sharing visibility."""
    controller = _controller({"m_priv": {CONF_OWNER: BOB, CONF_SHARED: False}})
    admin = _make_user(ALICE, role=UserRole.ADMIN)
    assert controller.provider_visible_to_user(_make_prov("m_priv"), admin) is True


def test_internal_call_sees_every_provider() -> None:
    """An internal/unauthenticated call (user is None) sees every provider."""
    controller = _controller({"m_priv": {CONF_OWNER: BOB, CONF_SHARED: False}})
    assert controller.provider_visible_to_user(_make_prov("m_priv"), None) is True


def test_unowned_provider_is_visible_to_everyone() -> None:
    """A provider without an owner set is a house provider and visible to all users."""
    controller = _controller({"m_house": {CONF_OWNER: "", CONF_SHARED: False}})
    assert controller.provider_visible_to_user(_make_prov("m_house"), _make_user(ALICE)) is True


def test_owner_sees_own_private_provider() -> None:
    """The owner always sees their own provider, even when it is not shared."""
    controller = _controller({"m_priv": {CONF_OWNER: BOB, CONF_SHARED: False}})
    assert controller.provider_visible_to_user(_make_prov("m_priv"), _make_user(BOB)) is True


def test_non_owner_cannot_see_private_owned_provider() -> None:
    """A non-owner cannot see a provider owned by someone else and not shared."""
    controller = _controller({"m_priv": {CONF_OWNER: BOB, CONF_SHARED: False}})
    assert controller.provider_visible_to_user(_make_prov("m_priv"), _make_user(ALICE)) is False


def test_non_owner_sees_shared_owned_provider() -> None:
    """A non-owner can see a provider owned by someone else but marked as shared."""
    controller = _controller({"m_shared": {CONF_OWNER: BOB, CONF_SHARED: True}})
    assert controller.provider_visible_to_user(_make_prov("m_shared"), _make_user(ALICE)) is True


def test_default_shared_install_shows_everything() -> None:
    """
    A fresh install stores no owner/shared values, so every provider stays visible.

    This preserves behavior for existing installs until someone assigns an owner and unticks
    ``shared`` - the config getter returns the ``shared`` default (True) and no owner.
    """
    controller = _controller({})  # nothing stored for any provider
    assert controller.provider_visible_to_user(_make_prov("m_a"), _make_user(ALICE)) is True


def test_non_music_provider_always_visible() -> None:
    """Ownership/sharing only applies to music providers; other types are always visible."""
    controller = _controller({"meta": {CONF_OWNER: BOB, CONF_SHARED: False}})
    metadata = _make_prov("meta", ProviderType.METADATA)
    assert controller.provider_visible_to_user(metadata, _make_user(ALICE)) is True


def test_apply_user_provider_filter_uses_ownership(monkeypatch) -> None:
    """_apply_user_provider_filter drops music providers not visible to the session user."""
    controller = _controller(
        {
            "m_house": {},  # unowned -> visible
            "m_priv": {CONF_OWNER: BOB, CONF_SHARED: False},  # private, not alice's
            "m_shared": {CONF_OWNER: BOB, CONF_SHARED: True},  # shared -> visible
        }
    )
    monkeypatch.setattr(
        "music_assistant.controllers.music.controller.get_current_user",
        lambda: _make_user(ALICE),
    )
    providers = [_make_prov("m_house"), _make_prov("m_priv"), _make_prov("m_shared")]
    result = controller._apply_user_provider_filter(providers)
    assert [p.instance_id for p in result] == ["m_house", "m_shared"]


def test_get_visible_provider_instance_ids_and_sees_all() -> None:
    """The visible-id set and the sees-all fast-path agree with per-provider visibility."""
    controller = _controller(
        {
            "m_house": {},
            "m_priv": {CONF_OWNER: BOB, CONF_SHARED: False},
        }
    )
    controller.mass.providers = [
        _make_prov("m_house"),
        _make_prov("m_priv"),
        _make_prov("meta", ProviderType.METADATA),
    ]
    alice = _make_user(ALICE)
    assert controller.get_visible_provider_instance_ids(alice) == {"m_house"}
    assert controller.user_sees_all_providers(alice) is False
    # bob owns m_priv and every other provider is unowned -> bob sees all music providers
    assert controller.user_sees_all_providers(_make_user(BOB)) is True
    # admins short-circuit to "sees all"
    assert controller.user_sees_all_providers(_make_user(ALICE, role=UserRole.ADMIN)) is True
