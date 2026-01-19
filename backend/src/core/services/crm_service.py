"""
CRMService
==========

Servicio de CRM para campañas promocionales (cumpleaños e inactivos)
y overview de clientes, integrado con el esquema unificado de
Supabase/PostgreSQL.

Se apoya en las tablas:

- public."customers"
- public."reservations"
- public."reservation_items"
- public."menu_items"
- public."preferences"
- public."customer_preferences"
- public."allergies"
- public."customer_allergies"

y en variables SMTP para enviar correos.
"""

import os
from datetime import datetime, date
from typing import List, Dict, Any, Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


class CRMService:
    """Servicio de CRM para campañas de marketing y vista general de clientes."""

    def __init__(self) -> None:
        load_dotenv()

        pg_conn = (
            os.getenv("SUPABASE_PG_CONN")
            or os.getenv("SUPABASE_SP_CONN")
            or os.getenv("SUPABASE_DB_URL")
            or os.getenv("DATABASE_URL")
        )
        if not pg_conn:
            raise RuntimeError(
                "CRMService: falta cadena de conexión. "
                "Configura SUPABASE_PG_CONN, SUPABASE_SP_CONN o DATABASE_URL en tu .env"
            )

        # Engine independiente, de solo lectura / marketing
        self.engine = create_engine(pg_conn, pool_pre_ping=True, future=True)

    # ------------------------------------------------------------------
    # Carga de tablas en memoria
    # ------------------------------------------------------------------
    def _load_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Carga las tablas relevantes para el CRM desde la base de datos.
        """
        with self.engine.connect() as conn:
            customers = pd.read_sql(
                text(
                    'SELECT customer_id, first_name, last_name, email, whatsapp, '
                    'birth_date, ticket_promedio_cliente '
                    'FROM public."customers"'
                ),
                conn,
            )

            reservations = pd.read_sql(
                text(
                    'SELECT reservation_id, customer_id, '
                    'COALESCE(start_iso, created_at) AS reservation_ts, '
                    "total_ticket "
                    'FROM public."reservations"'
                ),
                conn,
            )

            reservation_items = pd.read_sql(
                text(
                    'SELECT reservation_id, sku, quantity, price_at_visit '
                    'FROM public."reservation_items"'
                ),
                conn,
            )

            menu_items = pd.read_sql(
                text(
                    'SELECT sku, price '
                    'FROM public."menu_items"'
                ),
                conn,
            )

            # Preferencias / alergias
            preferences = pd.read_sql(
                text(
                    'SELECT preference_id, description '
                    'FROM public."preferences"'
                ),
                conn,
            )

            customer_preferences = pd.read_sql(
                text(
                    'SELECT customer_id, preference_id '
                    'FROM public."customer_preferences"'
                ),
                conn,
            )

            allergies = pd.read_sql(
                text(
                    'SELECT allergy_id, name '
                    'FROM public."allergies"'
                ),
                conn,
            )

            customer_allergies = pd.read_sql(
                text(
                    'SELECT customer_id, allergy_id '
                    'FROM public."customer_allergies"'
                ),
                conn,
            )

        return {
            "customers": customers,
            "reservations": reservations,
            "reservation_items": reservation_items,
            "menu_items": menu_items,
            "preferences": preferences,
            "customer_preferences": customer_preferences,
            "allergies": allergies,
            "customer_allergies": customer_allergies,
        }

    # ------------------------------------------------------------------
    # Cálculo de ticket promedio
    # ------------------------------------------------------------------
    def _compute_average_ticket(
        self,
        reservations: pd.DataFrame,
        reservation_items: pd.DataFrame,
        menu_items: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calcula el ticket promedio por cliente.

        Devuelve un DataFrame con columnas:
            - customer_id
            - average_ticket
        """
        # Si la tabla reservations ya tiene total_ticket, usamos eso
        if "total_ticket" in reservations.columns and not reservations.empty:
            df = reservations.copy()
            # Por seguridad, convertimos a float
            df["reservation_total"] = pd.to_numeric(
                df["total_ticket"], errors="coerce"
            ).fillna(0.0)
        else:
            if reservation_items.empty:
                return pd.DataFrame(columns=["customer_id", "average_ticket"])

            items = reservation_items.merge(menu_items, on="sku", how="left")
            items["total"] = items["quantity"] * items["price"]
            spend_per_reservation = (
                items.groupby("reservation_id")["total"]
                .sum()
                .reset_index(name="reservation_total")
            )
            df = reservations.merge(
                spend_per_reservation, on="reservation_id", how="left"
            )

        customer_spend = (
            df.groupby("customer_id")["reservation_total"]
            .mean()
            .reset_index()
        )
        customer_spend.columns = ["customer_id", "average_ticket"]
        return customer_spend

    # ------------------------------------------------------------------
    # Cumpleaños próximos
    # ------------------------------------------------------------------
    def _upcoming_birthdays(
        self, customers: pd.DataFrame, days_ahead: int
    ) -> pd.DataFrame:
        """
        Devuelve DataFrame con clientes cuyo cumpleaños cae dentro de
        los próximos `days_ahead` días.
        """
        if customers.empty:
            return pd.DataFrame()

        today = date.today()
        upcoming: List[Dict[str, Any]] = []

        for _, row in customers.iterrows():
            raw = row.get("birth_date")
            if not raw:
                continue

            try:
                bdate = pd.to_datetime(raw).date()
            except Exception:
                continue

            next_birthday = bdate.replace(year=today.year)
            if next_birthday < today:
                next_birthday = next_birthday.replace(year=today.year + 1)

            days_until = (next_birthday - today).days
            if 0 <= days_until <= days_ahead:
                upcoming.append(
                    {
                        "customer_id": row["customer_id"],
                        "name": f"{row.get('first_name', '')} {row.get('last_name', '')}".strip(),
                        "email": row.get("email"),
                        "days_until": int(days_until),
                    }
                )

        if not upcoming:
            return pd.DataFrame()
        return pd.DataFrame(upcoming)

    # ------------------------------------------------------------------
    # Clientes inactivos
    # ------------------------------------------------------------------
    def _inactive_customers(
        self, customers: pd.DataFrame, reservations: pd.DataFrame, days_since: int
    ) -> pd.DataFrame:
        """
        Devuelve clientes que no han visitado en más de `days_since` días.
        """
        today = datetime.now()
        if customers.empty:
            return pd.DataFrame()

        if reservations.empty:
            df = customers.copy()
            df["days_since_last"] = days_since + 1
            return df

        last_visit = (
            reservations.groupby("customer_id")["reservation_ts"]
            .max()
            .reset_index()
        )
        last_visit["days_since_last"] = (
            today - pd.to_datetime(last_visit["reservation_ts"])
        ).dt.days

        merged = customers.merge(last_visit, on="customer_id", how="left")
        merged["days_since_last"] = merged["days_since_last"].fillna(days_since + 1)

        return merged[merged["days_since_last"] > days_since]

    # ------------------------------------------------------------------
    # Campañas generadas (sin enviar)
    # ------------------------------------------------------------------
    def prepare_birthday_campaigns(self, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """
        Genera campañas de cumpleaños con descuento sugerido según ticket promedio.
        """
        tables = self._load_tables()
        customers = tables["customers"]

        customer_spend = self._compute_average_ticket(
            tables["reservations"],
            tables["reservation_items"],
            tables["menu_items"],
        )

        upcoming = self._upcoming_birthdays(customers, days_ahead)
        if upcoming.empty:
            return []

        campaigns: List[Dict[str, Any]] = []

        for _, row in upcoming.iterrows():
            cust_id = row["customer_id"]
            spend_row = customer_spend[
                customer_spend["customer_id"] == cust_id
            ]
            avg_ticket = (
                float(spend_row["average_ticket"].iloc[0])
                if not spend_row.empty
                else 0.0
            )

            if avg_ticket >= 300:
                discount = 20
            elif avg_ticket >= 150:
                discount = 15
            else:
                discount = 10

            campaigns.append(
                {
                    "customer_id": int(cust_id),
                    "name": row["name"],
                    "email": row.get("email"),
                    "days_until": int(row["days_until"]),
                    "suggested_discount": int(discount),
                    "message": (
                        f"Hola {row['name']}, tu cumpleaños está en {int(row['days_until'])} días. "
                        f"¡Te ofrecemos un {discount}% de descuento en tu próxima visita!"
                    ),
                }
            )

        return campaigns

    def prepare_inactive_campaigns(
        self, days_since: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Genera campañas para clientes inactivos (no han visitado en > days_since días).
        """
        tables = self._load_tables()
        customers = tables["customers"]
        reservations = tables["reservations"]

        inactives = self._inactive_customers(customers, reservations, days_since)
        if inactives.empty:
            return []

        campaigns: List[Dict[str, Any]] = []

        for _, row in inactives.iterrows():
            name = f"{row.get('first_name', '')} {row.get('last_name', '')}".strip()
            campaigns.append(
                {
                    "customer_id": int(row["customer_id"]),
                    "name": name,
                    "email": row.get("email"),
                    "days_since_last": int(row["days_since_last"]),
                    "message": (
                        f"Hola {name}, hace {int(row['days_since_last'])} días que no te vemos. "
                        "¡Te ofrecemos un 15% de descuento si reservas esta semana!"
                    ),
                }
            )

        return campaigns

    # ------------------------------------------------------------------
    # Overview de clientes
    # ------------------------------------------------------------------
    def get_customers_overview(self, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Devuelve una lista de clientes con métricas agregadas:

        - customer_id
        - name
        - email
        - whatsapp
        - birth_date
        - visits_count
        - last_visit
        - average_ticket
        - preferences (lista de descripciones)
        - allergies (lista de nombres)
        """
        tables = self._load_tables()

        customers = tables["customers"]
        reservations = tables["reservations"]
        reservation_items = tables["reservation_items"]
        menu_items = tables["menu_items"]
        preferences = tables["preferences"]
        customer_preferences = tables["customer_preferences"]
        allergies = tables["allergies"]
        customer_allergies = tables["customer_allergies"]

        # Métricas de visitas / último visit
        if reservations.empty:
            visits = pd.DataFrame(
                columns=["customer_id", "visits_count", "last_visit"]
            )
        else:
            df = reservations.copy()
            df["reservation_ts"] = pd.to_datetime(df["reservation_ts"])
            agg = (
                df.groupby("customer_id")["reservation_ts"]
                .agg(["count", "max"])
                .reset_index()
            )
            agg.columns = ["customer_id", "visits_count", "last_visit"]
            visits = agg

        # Ticket promedio
        avg_ticket_df = self._compute_average_ticket(
            reservations, reservation_items, menu_items
        )

        # Preferencias por cliente
        if customer_preferences.empty or preferences.empty:
            prefs_per_customer = pd.DataFrame(
                columns=["customer_id", "preferences"]
            )
        else:
            prefs = customer_preferences.merge(
                preferences, on="preference_id", how="left"
            )
            prefs_grouped = (
                prefs.groupby("customer_id")["description"]
                .apply(lambda x: [d for d in x.dropna().tolist()])
                .reset_index()
            )
            prefs_grouped.columns = ["customer_id", "preferences"]
            prefs_per_customer = prefs_grouped

        # Alergias por cliente
        if customer_allergies.empty or allergies.empty:
            allergies_per_customer = pd.DataFrame(
                columns=["customer_id", "allergies"]
            )
        else:
            alls = customer_allergies.merge(
                allergies, on="allergy_id", how="left"
            )
            alls_grouped = (
                alls.groupby("customer_id")["name"]
                .apply(lambda x: [d for d in x.dropna().tolist()])
                .reset_index()
            )
            alls_grouped.columns = ["customer_id", "allergies"]
            allergies_per_customer = alls_grouped

        # Merge maestro
        base = customers.copy()
        if not visits.empty:
            base = base.merge(visits, on="customer_id", how="left")
        if not avg_ticket_df.empty:
            base = base.merge(avg_ticket_df, on="customer_id", how="left")
        if not prefs_per_customer.empty:
            base = base.merge(prefs_per_customer, on="customer_id", how="left")
        if not allergies_per_customer.empty:
            base = base.merge(allergies_per_customer, on="customer_id", how="left")

        # Limpieza / defaults
        base["visits_count"] = base["visits_count"].fillna(0).astype(int)
        base["average_ticket"] = pd.to_numeric(
            base["average_ticket"], errors="coerce"
        ).fillna(0.0)
        base["preferences"] = base["preferences"].apply(
            lambda x: x if isinstance(x, list) else []
        )
        base["allergies"] = base["allergies"].apply(
            lambda x: x if isinstance(x, list) else []
        )

        # Ordenar por visitas / última visita
        if "last_visit" in base.columns:
            base = base.sort_values(
                by=["visits_count", "last_visit"], ascending=[False, False]
            )
        else:
            base = base.sort_values(by=["visits_count"], ascending=False)

        if limit and limit > 0:
            base = base.head(limit)

        result: List[Dict[str, Any]] = []
        for _, row in base.iterrows():
            first = row.get("first_name") or ""
            last = row.get("last_name") or ""
            name = f"{first} {last}".strip() or "Cliente sin nombre"
            last_visit = row.get("last_visit")
            if pd.isna(last_visit):
                last_visit_str: Optional[str] = None
            else:
                if isinstance(last_visit, (datetime, pd.Timestamp)):
                    last_visit_str = last_visit.isoformat()
                else:
                    last_visit_str = str(last_visit)

            birth_raw = row.get("birth_date")
            birth_str: Optional[str]
            if pd.isna(birth_raw):
                birth_str = None
            else:
                try:
                    birth_str = pd.to_datetime(birth_raw, dayfirst=True).date().isoformat()
                except Exception:
                    birth_str = str(birth_raw)

            result.append(
                {
                    "customer_id": int(row["customer_id"]),
                    "name": name,
                    "email": row.get("email"),
                    "whatsapp": row.get("whatsapp"),
                    "birth_date": birth_str,
                    "visits_count": int(row["visits_count"]),
                    "last_visit": last_visit_str,
                    "average_ticket": float(row["average_ticket"]),
                    "preferences": row["preferences"],
                    "allergies": row["allergies"],
                }
            )

        return result

    # ------------------------------------------------------------------
    # SMTP y envío de correos
    # ------------------------------------------------------------------
    def _send_email(self, to_address: str, subject: str, body: str) -> None:
        """Envía un correo usando las variables SMTP de entorno."""
        import smtplib
        from email.mime.text import MIMEText

        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USERNAME")
        smtp_password = os.getenv("SMTP_PASSWORD")
        email_from = os.getenv("EMAIL_FROM") or smtp_user

        if not all([smtp_server, smtp_user, smtp_password, email_from]):
            raise RuntimeError(
                "SMTP incompleto. Configura SMTP_SERVER, SMTP_USERNAME, "
                "SMTP_PASSWORD y EMAIL_FROM en tu .env"
            )

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = email_from
        msg["To"] = to_address

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)

    def send_birthday_campaigns(self, days_ahead: int = 7) -> Dict[str, Any]:
        """
        Envía por correo las campañas de cumpleaños de los próximos `days_ahead` días.
        """
        campaigns = self.prepare_birthday_campaigns(days_ahead)
        if not campaigns:
            return {"sent": 0, "planned": 0, "result": "No hay campañas para enviar."}

        sent = 0
        for c in campaigns:
            if not c.get("email"):
                continue
            try:
                self._send_email(
                    c["email"],
                    "¡Feliz cumpleaños anticipado!",
                    c["message"],
                )
                sent += 1
            except Exception:
                # No abortar la campaña por un fallo individual
                continue

        return {"sent": sent, "planned": len(campaigns)}

    def send_inactive_campaigns(self, days_since: int = 30) -> Dict[str, Any]:
        """
        Envía por correo campañas a clientes inactivos (> days_since días sin visitar).
        """
        campaigns = self.prepare_inactive_campaigns(days_since)
        if not campaigns:
            return {"sent": 0, "planned": 0, "result": "No hay campañas para enviar."}

        sent = 0
        for c in campaigns:
            if not c.get("email"):
                continue
            try:
                self._send_email(
                    c["email"],
                    "¡Te extrañamos!",
                    c["message"],
                )
                sent += 1
            except Exception:
                continue

        return {"sent": sent, "planned": len(campaigns)}

    # ------------------------------------------------------------------
    # Checks de infraestructura
    # ------------------------------------------------------------------
    def db_check(self) -> Dict[str, Any]:
        """Verifica conexión a la base de datos usada por el CRM."""
        try:
            with self.engine.connect() as conn:
                row = conn.execute(
                    text("select current_database(), now()")
                ).fetchone()
            return {"ok": True, "database": row[0], "time": str(row[1])}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def smtp_check(self) -> Dict[str, Any]:
        """Prueba login SMTP (sin enviar correo real)."""
        import smtplib

        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USERNAME")
        smtp_password = os.getenv("SMTP_PASSWORD")

        try:
            with smtplib.SMTP(smtp_server, smtp_port, timeout=20) as s:
                s.starttls()
                s.login(smtp_user, smtp_password)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def send_test_email(self, to_address: str) -> Dict[str, Any]:
        """Envía un correo de prueba simple para validar SMTP."""
        try:
            self._send_email(
                to_address,
                "Prueba SMTP · Solertia CRM",
                "Si ves este correo, el SMTP está funcionando ✅",
            )
            return {"ok": True, "to": to_address}
        except Exception as e:
            return {"ok": False, "error": str(e)}
