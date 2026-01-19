import React, { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Calendar, Clock, Users, Phone, Search } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";

const API_BASE_URL =
  import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const GOOGLE_CALENDAR_EMBED_URL =
  import.meta.env.VITE_GOOGLE_CALENDAR_URL ??
  "https://calendar.google.com/calendar/embed?src=solertia.grp%40gmail.com&ctz=America%2FMonterrey";

interface Reservation {
  id: number;
  customer_id: number;
  date: string;
  time: string;
  guests: number;
  status: string;
  name?: string;
  phone?: string;
  notes?: string;
  table_number?: string;
}

const Reservations: React.FC = () => {
  const [reservations, setReservations] = useState<Reservation[]>([]);
  const [searchCustomerId, setSearchCustomerId] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!searchCustomerId.trim()) {
      toast.error("Ingresa un customer_id para buscar.");
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const response = await fetch(
        `${API_BASE_URL}/reservations/${searchCustomerId.trim()}`
      );

      if (!response.ok) {
        throw new Error("No se pudieron obtener las reservaciones.");
      }

      const data: Reservation[] = await response.json();
      setReservations(data);

      if (data.length === 0) {
        toast("No se encontraron reservaciones para ese customer_id.");
      } else {
        toast.success("Reservaciones cargadas correctamente.");
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Error desconocido al cargar reservaciones.";
      setError(message);
      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

  const totalReservations = reservations.length;
  const confirmedCount = reservations.filter(
    (r) => r.status?.toLowerCase() === "confirmed"
  ).length;
  const pendingCount = reservations.filter(
    (r) => r.status?.toLowerCase() === "pending"
  ).length;
  const totalGuests = reservations.reduce(
    (acc, r) => acc + (r.guests ?? 0),
    0
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold tracking-tight">Reservations</h1>
          <p className="text-muted-foreground mt-2">
            Manage and track restaurant reservations (conectado a{" "}
            <code>/reservations/&#123;customer_id&#125;</code> en tu API de Python).
          </p>
        </div>
      </div>

      {/* Search */}
      <Card>
        <CardHeader>
          <CardTitle>Buscar reservaciones</CardTitle>
          <CardDescription>
            Ingresa un <code>customer_id</code> para cargar sus reservaciones desde el backend.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-3 md:flex-row md:items-center">
            <div className="flex-1 flex items-center gap-2">
              <Search className="w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="customer_id (por ejemplo, 1)"
                value={searchCustomerId}
                onChange={(e) => setSearchCustomerId(e.target.value)}
              />
            </div>
            <Button onClick={handleSearch} disabled={loading}>
              {loading ? "Cargando..." : "Buscar"}
            </Button>
          </div>
          {error && (
            <p className="mt-2 text-sm text-red-500">
              {error}
            </p>
          )}
        </CardContent>
      </Card>

      {/* Summary Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Total Reservations
            </CardTitle>
            <Calendar className="w-4 h-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalReservations}</div>
            <p className="text-xs text-muted-foreground">
              Reservaciones cargadas para el customer_id actual.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Confirmed
            </CardTitle>
            <Badge variant="outline" className="text-xs">
              OK
            </Badge>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{confirmedCount}</div>
            <p className="text-xs text-muted-foreground">
              Reservaciones confirmadas.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Pending
            </CardTitle>
            <Badge variant="outline" className="text-xs">
              ...
            </Badge>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{pendingCount}</div>
            <p className="text-xs text-muted-foreground">
              Reservaciones en espera de confirmación.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Total Guests
            </CardTitle>
            <Users className="w-4 h-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalGuests}</div>
            <p className="text-xs text-muted-foreground">
              Número total de comensales en las reservaciones cargadas.
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Google Calendar */}
      <Card>
        <CardHeader>
          <CardTitle>Calendario de Reservaciones</CardTitle>
          <CardDescription>
            Vista pública del calendario de Google con las reservaciones (solertia.grp@gmail.com).
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="w-full overflow-hidden rounded-xl border">
            <iframe
              src={GOOGLE_CALENDAR_EMBED_URL}
              style={{ border: 0 }}
              className="w-full h-[600px]"
              loading="lazy"
              referrerPolicy="no-referrer-when-downgrade"
            />
          </div>
        </CardContent>
      </Card>

      {/* Reservations Table */}
      <Card>
        <CardHeader>
          <CardTitle>Reservaciones</CardTitle>
          <CardDescription>
            Datos retornados por tu endpoint{" "}
            <code>/reservations/&#123;customer_id&#125;</code>.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>ID</TableHead>
                <TableHead>Fecha</TableHead>
                <TableHead>Hora</TableHead>
                <TableHead>Clientes</TableHead>
                <TableHead>Estado</TableHead>
                <TableHead>Nombre</TableHead>
                <TableHead>Teléfono</TableHead>
                <TableHead>Mesa</TableHead>
                <TableHead>Notas</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {reservations.length === 0 ? (
                <TableRow>
                  <TableCell
                    colSpan={9}
                    className="text-center text-sm text-muted-foreground"
                  >
                    No reservations loaded. Search by a customer_id above.
                  </TableCell>
                </TableRow>
              ) : (
                reservations.map((r) => (
                  <TableRow key={r.id}>
                    <TableCell>{r.id}</TableCell>
                    <TableCell>{r.date}</TableCell>
                    <TableCell className="whitespace-nowrap">
                      <div className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        <span>{r.time}</span>
                      </div>
                    </TableCell>
                    <TableCell>{r.guests}</TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          r.status?.toLowerCase() === "confirmed"
                            ? "default"
                            : r.status?.toLowerCase() === "cancelled"
                            ? "destructive"
                            : "outline"
                        }
                      >
                        {r.status}
                      </Badge>
                    </TableCell>
                    <TableCell>{r.name ?? "-"}</TableCell>
                    <TableCell>
                      {r.phone ? (
                        <a
                          href={`tel:${r.phone}`}
                          className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
                        >
                          <Phone className="w-3 h-3" />
                          {r.phone}
                        </a>
                      ) : (
                        "-"
                      )}
                    </TableCell>
                    <TableCell>{r.table_number ?? "-"}</TableCell>
                    <TableCell className="max-w-xs truncate">
                      {r.notes ?? "-"}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
};

export default Reservations;
