"""Setup flow for the AmpliPi provider."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import ConfigEntryType

from music_assistant.helpers.util import get_primary_ip_address_from_zeroconf
from music_assistant.models.setup_flow import SetupFlowError
from music_assistant.providers.amplipi.constants import (
    CONF_HOST,
    DEFAULT_HOST,
    MDNS_NAME,
    MDNS_TYPE,
)

if TYPE_CHECKING:
    from music_assistant.models.setup_flow import SetupSession

LOGGER = logging.getLogger(__name__)

# how long the form waits for an AmpliPi to answer on mDNS before falling back to the
# default hostname; kept short so setup does not appear to hang on a network without one.
_DISCOVERY_TIMEOUT = 3.0

_ENTRIES = (
    ConfigEntry(
        key=CONF_HOST,
        type=ConfigEntryType.STRING,
        required=True,
    ),
)


async def run_setup(session: SetupSession) -> None:
    """Run the setup flow: collect the connection details and create the provider."""
    errors: dict[str, str] | None = None
    setup_data = dict(session.context.setup_data)
    if CONF_HOST not in setup_data:
        setup_data[CONF_HOST] = await _discover_host(session)
    while True:
        entries = [
            replace(entry, value=setup_data.get(entry.key, entry.value)) for entry in _ENTRIES
        ]
        submitted = await session.form(entries, step_id="user", errors=errors, last_step=True)
        setup_data.update(submitted)
        try:
            await session.finish(setup_data)
            return
        except SetupFlowError as err:
            errors = {"base": err.translation_key or str(err)}


async def _discover_host(session: SetupSession) -> str:
    """
    Return the address to prefill the host field with.

    Prefers the hostname an AmpliPi found on the network advertises, as its IP address is
    usually a DHCP lease that stops working once it is reassigned. The IP is used only
    when this host cannot resolve that name (mDNS resolution is not available
    everywhere Music Assistant runs). The value is only a prefill: the user can always
    point the provider at another controller.
    """
    discovery_info = await session.mass.discovery.async_find_mdns_service(
        MDNS_TYPE, MDNS_NAME, timeout=_DISCOVERY_TIMEOUT
    )
    if discovery_info is None:
        LOGGER.debug("No %s service found on mDNS, offering %s", MDNS_NAME, DEFAULT_HOST)
        return DEFAULT_HOST
    hostname = (discovery_info.server or "").rstrip(".")
    if hostname and await _is_resolvable(hostname):
        LOGGER.debug("Discovered AmpliPi at %s", hostname)
        return hostname
    address = get_primary_ip_address_from_zeroconf(discovery_info)
    LOGGER.debug(
        "Discovered AmpliPi advertising %s, which does not resolve here; offering %s",
        hostname or "no hostname",
        address or DEFAULT_HOST,
    )
    return address or DEFAULT_HOST


async def _is_resolvable(hostname: str) -> bool:
    """Return whether the host running Music Assistant can resolve the given hostname."""
    try:
        await asyncio.get_running_loop().getaddrinfo(hostname, None)
    except OSError:
        return False
    return True
