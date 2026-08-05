import time

from lumen.force.mnemonic.event_buffer import Event, EventMemoryBuffer


def test_append_adds_event():
    buf = EventMemoryBuffer(max_events=100, max_age_hours=24.0)
    event = Event(raw_text="hello", source="user", session_id="s1")
    buf.append(event)
    assert len(buf._buffer) == 1
    assert buf._buffer[0].raw_text == "hello"


def test_append_multiple_events():
    buf = EventMemoryBuffer(max_events=100, max_age_hours=24.0)
    for i in range(5):
        buf.append(Event(raw_text=f"event {i}", source="user"))
    assert len(buf._buffer) == 5


def test_query_since_returns_recent_events():
    buf = EventMemoryBuffer(max_events=100, max_age_hours=24.0)
    now = time.time()
    e1 = Event(raw_text="old", source="user")
    e1.timestamp = now - 1000
    e2 = Event(raw_text="new", source="user")
    e2.timestamp = now - 10
    buf.append(e1)
    buf.append(e2)
    recent = buf.query_since(now - 50)
    assert len(recent) == 1
    assert recent[0].raw_text == "new"


def test_drain_expired_removes_old_events():
    buf = EventMemoryBuffer(max_events=100, max_age_hours=1.0)
    now = time.time()
    e1 = Event(raw_text="stale", source="user")
    e1.timestamp = now - 7200
    e2 = Event(raw_text="fresh", source="user")
    e2.timestamp = now - 10
    buf.append(e1)
    buf.append(e2)
    expired = buf.drain_expired(cutoff_timestamp=now - 3600)
    assert len(expired) == 1
    assert expired[0].raw_text == "stale"
    assert len(buf._buffer) == 1
    assert buf._buffer[0].raw_text == "fresh"


def test_drain_expired_default_cutoff():
    buf = EventMemoryBuffer(max_events=100, max_age_hours=24.0)
    now = time.time()
    e = Event(raw_text="very old", source="user")
    e.timestamp = now - 100_000
    buf.append(e)
    expired = buf.drain_expired()
    assert len(expired) == 1
    assert len(buf._buffer) == 0


def test_max_events_respected():
    buf = EventMemoryBuffer(max_events=3, max_age_hours=24.0)
    for i in range(5):
        buf.append(Event(raw_text=f"event {i}", source="user"))
    assert len(buf._buffer) == 3
