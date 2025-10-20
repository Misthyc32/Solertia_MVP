# Test file - simplified for basic functionality testing
from db import init_db, SessionLocal
from graph import _normalize_date_es, _combine_date_time_to_rfc3339, _validate_reservation
from config import TZ

def test_basic_functionality():
    """Test basic reservation functionality"""
    # Initialize DB
    init_db()
    db = SessionLocal()
    
    # Test data
    appt = {
        "name": "Test User",
        "date": "mañana", 
        "time_start": "20:00",
        "time_end": "22:00",
        "people": 4
    }
    
    # Test normalization
    appt["date"] = _normalize_date_es(appt["date"], tz=TZ)
    _validate_reservation(appt, tz=TZ)
    
    # Test ISO conversion
    start_iso = _combine_date_time_to_rfc3339(appt["date"], appt["time_start"], tz=TZ)
    end_iso = _combine_date_time_to_rfc3339(appt["date"], appt["time_end"], tz=TZ)
    
    print("✅ Basic functionality test passed")
    print(f"Normalized date: {appt['date']}")
    print(f"Start ISO: {start_iso}")
    print(f"End ISO: {end_iso}")
    
    db.close()

if __name__ == "__main__":
    test_basic_functionality()
