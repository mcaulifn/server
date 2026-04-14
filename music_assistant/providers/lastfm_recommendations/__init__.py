"""Last.fm Recommendations music provider for Music Assistant."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from music_assistant_models.config_entries import (
    ConfigEntry,
    ConfigValueOption,
    ConfigValueType,
)
from music_assistant_models.enums import ConfigEntryType, ProviderFeature
from music_assistant_models.errors import MusicAssistantError
from music_assistant_models.media_items import RecommendationFolder  # noqa: TC002

from music_assistant.models.music_provider import MusicProvider
from music_assistant.providers.lastfm_recommendations.api_client import LastFMAPIClient
from music_assistant.providers.lastfm_recommendations.mbid_resolver import MBIDResolver
from music_assistant.providers.lastfm_recommendations.recommendations import (
    LastFMRecommendationManager,
)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant


SUPPORTED_FEATURES = {
    ProviderFeature.RECOMMENDATIONS,
}

# Config action constants
CONF_ACTION_CLEAR_CACHE = "clear_cache"

# Curated list of popular countries for Last.fm geo charts
# Last.fm API expects full country names (not ISO codes)
# This list covers major music markets and can be expanded based on user requests
GEO_COUNTRIES = [
    "Argentina",
    "Australia",
    "Austria",
    "Belgium",
    "Brazil",
    "Canada",
    "China",
    "Czech Republic",
    "Denmark",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Iceland",
    "India",
    "Ireland",
    "Israel",
    "Italy",
    "Japan",
    "Lithuania",
    "Mexico",
    "Netherlands",
    "New Zealand",
    "Norway",
    "Philippines",
    "Poland",
    "Portugal",
    "Serbia",
    "Singapore",
    "Slovenia",
    "South Africa",
    "South Korea",
    "Spain",
    "Sweden",
    "Switzerland",
    "Thailand",
    "Turkey",
    "Ukraine",
    "United Arab Emirates",
    "United Kingdom",
    "United States",
]


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> LastFMRecommendationsProvider:
    """Initialize provider(instance) with given configuration."""
    return LastFMRecommendationsProvider(mass, manifest, config, SUPPORTED_FEATURES)


async def get_config_entries(
    mass: MusicAssistant,
    instance_id: str | None = None,
    action: str | None = None,
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """Return Config entries to setup this provider."""
    if action == CONF_ACTION_CLEAR_CACHE and instance_id:
        provider = mass.get_provider(instance_id)
        if isinstance(provider, LastFMRecommendationsProvider):
            await provider.recommendations_manager.clear_cache()
            mass.create_task(provider._populate_recommendations())

    return (
        ConfigEntry(
            key="api_key",
            type=ConfigEntryType.SECURE_STRING,
            label="Last.fm API Key",
            required=True,
            description="Get your API key from https://www.last.fm/api/account/create",
            value=values.get("api_key") if values else None,
        ),
        ConfigEntry(
            key="username",
            type=ConfigEntryType.STRING,
            label="Last.fm Username",
            required=False,
            description="Your Last.fm username for genre-based recommendations (optional)",
            value=values.get("username") if values else None,
        ),
        ConfigEntry(
            key="refresh_interval",
            type=ConfigEntryType.INTEGER,
            label="Refresh Interval (hours)",
            default_value=6,
            description="How often to refresh recommendations (0 to disable automatic refresh)",
            category="recommendations",
            range=(0, 168),  # 0 to 1 week
        ),
        ConfigEntry(
            key="enable_similar_artists",
            type=ConfigEntryType.BOOLEAN,
            label="Enable Similar Artists (Personalized)",
            default_value=False,
            description="Show similar artists based on your listening history",
            category="recommendations",
        ),
        ConfigEntry(
            key="enable_similar_tracks",
            type=ConfigEntryType.BOOLEAN,
            label="Enable Similar Tracks (Personalized)",
            default_value=False,
            description="Show similar tracks based on your listening history",
            category="recommendations",
        ),
        ConfigEntry(
            key="enable_top_artists",
            type=ConfigEntryType.BOOLEAN,
            label="Enable Global Top Artists",
            default_value=False,
            description="Show worldwide top artists chart from Last.fm",
            category="recommendations",
        ),
        ConfigEntry(
            key="enable_top_tracks",
            type=ConfigEntryType.BOOLEAN,
            label="Enable Global Top Tracks",
            default_value=False,
            description="Show worldwide top tracks chart from Last.fm",
            category="recommendations",
        ),
        ConfigEntry(
            key="enable_genre_artists",
            type=ConfigEntryType.BOOLEAN,
            label="Enable Genre Artists",
            default_value=False,
            description="Show top artists from your most played genre (requires username)",
            category="recommendations",
        ),
        ConfigEntry(
            key="enable_genre_albums",
            type=ConfigEntryType.BOOLEAN,
            label="Enable Genre Albums",
            default_value=False,
            description="Show top albums from your most played genre (requires username)",
            category="recommendations",
        ),
        ConfigEntry(
            key="enable_genre_tracks",
            type=ConfigEntryType.BOOLEAN,
            label="Enable Genre Tracks",
            default_value=False,
            description="Show top tracks from your most played genre (requires username)",
            category="recommendations",
        ),
        ConfigEntry(
            key="geo_country",
            type=ConfigEntryType.STRING,
            label="Country for Geographic Charts",
            default_value="Argentina",
            description="Select country for geography-based top artists and tracks",
            options=[ConfigValueOption(country, country) for country in GEO_COUNTRIES],
            category="recommendations",
        ),
        ConfigEntry(
            key="enable_geo_artists",
            type=ConfigEntryType.BOOLEAN,
            label="Enable Geographic Top Artists",
            default_value=False,
            description="Show top artists from selected country",
            category="recommendations",
        ),
        ConfigEntry(
            key="enable_geo_tracks",
            type=ConfigEntryType.BOOLEAN,
            label="Enable Geographic Top Tracks",
            default_value=False,
            description="Show top tracks from selected country",
            category="recommendations",
        ),
        ConfigEntry(
            key=CONF_ACTION_CLEAR_CACHE,
            type=ConfigEntryType.ACTION,
            label="Clear Recommendation Cache",
            description=(
                "Clear all cached recommendations. "
                "Use if a provider was removed or recommendations are stale."
            ),
            action=CONF_ACTION_CLEAR_CACHE,
            action_label="Clear Cache",
            category="advanced",
            required=False,
        ),
    )


class LastFMRecommendationsProvider(MusicProvider):
    """Last.fm Recommendations Provider for Music Assistant."""

    async def handle_async_init(self) -> None:
        """Handle async initialization of the provider."""
        self.api = LastFMAPIClient(self)
        self.mbid_resolver = MBIDResolver(self)
        self.recommendations_manager = LastFMRecommendationManager(self)

        # Load cached folders so recommendations survive a provider reload.
        cache_key = f"recommendation_folders_{self.instance_id}"
        cached_folders = await self.mass.cache.get(cache_key)

        if cached_folders and isinstance(cached_folders, list):
            self._recommendation_folders: list[RecommendationFolder] = cached_folders
            self._recommendations_populated = True
            self.logger.debug("Loaded %d recommendation folders from cache", len(cached_folders))
        else:
            self._recommendation_folders = []
            self._recommendations_populated = False
            self.mass.create_task(self._populate_recommendations())

        self._schedule_refresh()

    async def _populate_recommendations(self) -> None:
        """Populate recommendation folders in the background."""
        try:
            # Wait for other providers (e.g. Spotify) to finish loading before resolving items,
            # otherwise resolution fails when no streaming providers are available yet.
            self.logger.debug(
                "Waiting 20 seconds for other providers to load before building recommendations"
            )
            await asyncio.sleep(20)

            self.logger.info("Building Last.fm recommendations")

            # Build folders incrementally so each category appears as soon as it's ready.
            personalized_folders = (
                await self.recommendations_manager._get_personalized_recommendations()
            )
            if personalized_folders:
                self._recommendation_folders.extend(personalized_folders)
                self.logger.debug(
                    "Added %d personalized recommendation folder(s)", len(personalized_folders)
                )

            global_folders = await self.recommendations_manager._get_global_recommendations()
            if global_folders:
                self._recommendation_folders.extend(global_folders)
                self.logger.debug("Added %d global recommendation folder(s)", len(global_folders))

            genre_folders = await self.recommendations_manager._get_genre_based_recommendations()
            if genre_folders:
                self._recommendation_folders.extend(genre_folders)
                self.logger.debug(
                    "Added %d genre-based recommendation folder(s)", len(genre_folders)
                )

            geo_folders = await self.recommendations_manager._get_geo_based_recommendations()
            if geo_folders:
                self._recommendation_folders.extend(geo_folders)
                self.logger.debug(
                    "Added %d geography-based recommendation folder(s)", len(geo_folders)
                )

            self._recommendations_populated = True
            self.logger.info(
                "Last.fm recommendations built (%d folders)",
                len(self._recommendation_folders),
            )

            cache_key = f"recommendation_folders_{self.instance_id}"
            await self.mass.cache.set(
                cache_key,
                self._recommendation_folders,
                expiration=60 * 60 * 24,
            )
        except MusicAssistantError as err:
            self.logger.warning("Failed to populate recommendations: %s", err)

    def _schedule_refresh(self) -> None:
        """Schedule the next periodic refresh of recommendations."""
        refresh_interval_value = self.config.get_value("refresh_interval")
        if isinstance(refresh_interval_value, (int, float)):
            refresh_interval_hours = int(refresh_interval_value)
        else:
            refresh_interval_hours = 6
        if refresh_interval_hours <= 0:
            self.logger.debug("Automatic refresh disabled (interval set to 0)")
            return

        refresh_interval_seconds = float(refresh_interval_hours * 3600)

        self.mass.call_later(
            refresh_interval_seconds,
            self._refresh_recommendations,
            task_id=f"lastfm_recommendations_refresh_{self.instance_id}",
        )
        self.logger.debug(
            "Scheduled next recommendations refresh in %d hours", refresh_interval_hours
        )

    async def _refresh_recommendations(self) -> None:
        """Re-populate recommendations and reschedule the next refresh."""
        try:
            self.logger.debug("Refreshing Last.fm recommendations (scheduled)")
            await self._populate_recommendations()
        except MusicAssistantError as err:
            self.logger.warning("Failed to refresh recommendations: %s", err)
        finally:
            self._schedule_refresh()

    async def recommendations(self) -> list[RecommendationFolder]:
        """Return this provider's recommendation folders.

        On first call (before background population completes) this returns an empty list.
        Subsequent calls return progressively more populated folders.
        """
        return self._recommendation_folders
