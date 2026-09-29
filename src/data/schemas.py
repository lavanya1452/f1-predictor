"""Canonical records for data returned by the Jolpica/Ergast API."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class CanonicalRecord(BaseModel):
    """Base type that rejects unreviewed fields in canonical records."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Circuit(CanonicalRecord):
    circuit_id: str
    name: str
    locality: str | None = None
    country: str | None = None
    latitude: float | None = None
    longitude: float | None = None


class Driver(CanonicalRecord):
    driver_id: str
    permanent_number: str | None = None
    code: str | None = None
    given_name: str
    family_name: str
    date_of_birth: date | None = None
    nationality: str | None = None


class Constructor(CanonicalRecord):
    constructor_id: str
    name: str
    nationality: str | None = None


class Race(CanonicalRecord):
    season: int = Field(ge=1950)
    round: int = Field(ge=1)
    race_id: str
    circuit_id: str
    name: str
    date: date
    time: str | None = None
    url: str | None = None


class QualifyingResult(CanonicalRecord):
    season: int = Field(ge=1950)
    round: int = Field(ge=1)
    race_id: str
    driver_id: str
    constructor_id: str
    position: int = Field(ge=1)
    q1: str | None = None
    q2: str | None = None
    q3: str | None = None


class RaceResult(CanonicalRecord):
    season: int = Field(ge=1950)
    round: int = Field(ge=1)
    race_id: str
    driver_id: str
    constructor_id: str
    grid: int | None = Field(default=None, ge=0)
    position: int | None = Field(default=None, ge=1)
    position_text: str
    points: float = Field(ge=0)
    laps: int = Field(ge=0)
    status: str
    fastest_lap_rank: int | None = Field(default=None, ge=1)


class DriverStanding(CanonicalRecord):
    season: int = Field(ge=1950)
    round: int = Field(ge=1)
    driver_id: str
    position: int = Field(ge=1)
    points: float = Field(ge=0)
    wins: int = Field(ge=0)


class ConstructorStanding(CanonicalRecord):
    season: int = Field(ge=1950)
    round: int = Field(ge=1)
    constructor_id: str
    position: int = Field(ge=1)
    points: float = Field(ge=0)
    wins: int = Field(ge=0)