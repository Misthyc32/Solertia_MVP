# API Integration Guide

This frontend connects to your Python FastAPI backend from both repositories.

## Backend Repositories

- **Solertia_MVP**: Main restaurant assistant API (chat, reservations, menu, manager analytics)
- **MVPFASE2**: CRM agent, trend algorithm, and database schemas

## API Endpoints to Connect

### 1. Chat Assistant (`/chat`)
**File**: `src/pages/Chat.tsx`

```typescript
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: userInput,
    customer_id: 'user_123',
    user_data: {
      phone: '+1234567890',
      name: 'User Name'
    }
  })
});
```

### 2. Reservations (`/reservations`)
**File**: `src/pages/Reservations.tsx`

```typescript
// Get reservations
const response = await fetch('http://localhost:8000/reservations/{customer_id}');

// Create reservation
const response = await fetch('http://localhost:8000/reservations', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    customer_id: 'user_123',
    date: '2025-11-10',
    time: '19:00',
    guests: 4
  })
});
```

### 3. Menu Search (`/menu/search`)
**File**: `src/pages/Menu.tsx`

```typescript
const response = await fetch('http://localhost:8000/menu/search', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: searchQuery,
    category: selectedCategory
  })
});
```

### 4. Manager Analytics (`/manager/ask`)
**File**: `src/pages/Analytics.tsx`

```typescript
// SQL Agent queries
const response = await fetch('http://localhost:8000/manager/ask', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: 'Top 10 SKUs por revenue para store_id=3 toda la historia'
  })
});

// Get store map data
const mapResponse = await fetch('http://localhost:8000/manager/map_data');

// Get plot visualization
const plotResponse = await fetch(`http://localhost:8000/manager/plots/${plot_id}`);
```

### 5. CRM System (from MVPFASE2)
**File**: `src/pages/CRM.tsx`

The CRM agent from the second repository should be integrated with endpoints like:

```typescript
// Get all customers with preferences and allergies
const response = await fetch('http://localhost:8000/crm/customers');

// Get customer details
const response = await fetch(`http://localhost:8000/crm/customers/{customer_id}`);

// Update customer preferences
const response = await fetch(`http://localhost:8000/crm/customers/{customer_id}/preferences`, {
  method: 'PUT',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    preferences: ['Vegetarian', 'No spicy'],
    allergies: ['Nuts']
  })
});
```

## Environment Setup

Create a `.env` file in the project root:

```env
VITE_API_URL=http://localhost:8000
```

Then use it in your code:

```typescript
const API_URL = import.meta.env.VITE_API_URL;
const response = await fetch(`${API_URL}/chat`, { ... });
```

## Features Integration Checklist

### From Solertia_MVP:
- ✅ Chat Interface - `/chat` endpoint
- ✅ Reservations - `/reservations` endpoints
- ✅ Menu Management - `/menu/*` endpoints
- ✅ Manager Analytics - `/manager/*` endpoints with SQL agent
- ✅ Store Map - `/manager/map_data` endpoint

### From MVPFASE2:
- ✅ CRM Agent - `crm_agent.py` integration needed
- ✅ Restaurant Agent - `restaurant_agent.py` integration needed
- ✅ Trend Algorithm - `trend_algorithm.py` for analytics
- ✅ Database Schema - Use provided CSVs and SQL schema

## Database Schema (from MVPFASE2)

The second repository includes comprehensive database schemas:

- `customers.csv` - Customer information
- `customer_preferences.csv` - Dietary preferences
- `customer_allergies.csv` - Allergy information
- `stores.csv` - Restaurant locations with coordinates
- `menu_items.csv` - Menu items
- `reservations.csv` - Reservation data
- `reservation_items.csv` - Ordered items per reservation
- `waiters.csv` - Staff information
- `central_dataset.csv` - Unified analytics data

## Next Steps

1. **Start your Python backend** from Solertia_MVP:
   ```bash
   cd Solertia_MVP
   python run.py
   ```

2. **Integrate CRM agent** from MVPFASE2 into your backend

3. **Update API calls** in the frontend pages to connect to your endpoints

4. **Test each feature** to ensure proper communication between frontend and backend

5. **Deploy** both frontend and backend when ready

## CORS Configuration

Make sure your FastAPI backend allows CORS:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
