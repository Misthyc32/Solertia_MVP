import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Users, Search, Mail, Phone, AlertTriangle, Star } from "lucide-react";
import { toast } from "sonner";

const API_BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

type Customer = {
  customer_id: number;
  name: string;
  email?: string | null;
  whatsapp?: string | null;
  birth_date?: string | null;
  visits_count: number;
  last_visit?: string | null;
  average_ticket: number;
  preferences: string[];
  allergies: string[];
};

const CRM = () => {
  const [customers, setCustomers] = useState<Customer[]>([]);
  const [search, setSearch] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  // Cargar clientes desde el backend
  useEffect(() => {
    const loadCustomers = async () => {
      setIsLoading(true);
      try {
        const res = await fetch(`${API_BASE_URL}/crm/customers?limit=200`);
        if (!res.ok) {
          const text = await res.text();
          throw new Error(text || `Error HTTP ${res.status}`);
        }
        const data: { customers: Customer[]; count: number } = await res.json();
        setCustomers(data.customers || []);
      } catch (err: any) {
        console.error(err);
        toast.error("Error al cargar los clientes del CRM.");
        setCustomers([]);
      } finally {
        setIsLoading(false);
      }
    };

    void loadCustomers();
  }, []);

  const filtered = useMemo(() => {
    if (!search.trim()) return customers;
    const q = search.toLowerCase();
    return customers.filter((c) => {
      return (
        c.name.toLowerCase().includes(q) ||
        (c.email || "").toLowerCase().includes(q) ||
        (c.whatsapp || "").toLowerCase().includes(q)
      );
    });
  }, [customers, search]);

  const totalCustomers = customers.length;
  const totalVisits = customers.reduce((acc, c) => acc + c.visits_count, 0);
  const avgTicketGlobal =
    customers.length > 0
      ? customers.reduce((acc, c) => acc + c.average_ticket, 0) / customers.length
      : 0;

  const customersWithAllergies = customers.filter((c) => c.allergies.length > 0).length;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-4xl font-bold tracking-tight">CRM</h1>
        <p className="text-muted-foreground mt-2">
          Overview de clientes, visitas y perfiles, conectado a tu backend en <code>/crm/customers</code>.
        </p>
      </div>

      {/* Resumen superior */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Total Customers
            </CardTitle>
            <Users className="h-4 w-4 text-primary" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalCustomers}</div>
            <p className="text-xs text-muted-foreground mt-1">
              En base de datos
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Total Visits
            </CardTitle>
            <Star className="h-4 w-4 text-primary" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalVisits}</div>
            <p className="text-xs text-muted-foreground mt-1">
              Suma de visitas registradas
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Avg Ticket (global)
            </CardTitle>
            <span className="text-xs font-mono text-primary">MXN</span>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              ${avgTicketGlobal.toFixed(2)}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Promedio entre todos los clientes
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Customers with Allergies
            </CardTitle>
            <AlertTriangle className="h-4 w-4 text-destructive" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{customersWithAllergies}</div>
            <p className="text-xs text-muted-foreground mt-1">
              Requieren atención especial
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Buscador */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center gap-3">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Buscar por nombre, email o whatsapp..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-10"
              />
            </div>
            {isLoading && (
              <span className="text-xs text-muted-foreground">
                Cargando clientes...
              </span>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Tabla de clientes */}
      <Card>
        <CardHeader>
          <CardTitle>Clientes</CardTitle>
          <CardDescription>
            Datos provenientes de tu base unificada (customers, reservations, preferences, allergies)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="max-h-[500px]">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Cliente</TableHead>
                  <TableHead>Contacto</TableHead>
                  <TableHead>Visitas</TableHead>
                  <TableHead>Última visita</TableHead>
                  <TableHead>Ticket promedio</TableHead>
                  <TableHead>Preferencias</TableHead>
                  <TableHead>Alergias</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filtered.map((c) => (
                  <TableRow key={c.customer_id}>
                    <TableCell className="font-medium">
                      <div className="flex flex-col">
                        <span>{c.name || `ID ${c.customer_id}`}</span>
                        {c.birth_date && (
                          <span className="text-xs text-muted-foreground">
                            Cumpleaños: {new Date(c.birth_date).toLocaleDateString()}
                          </span>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex flex-col gap-1 text-xs text-muted-foreground">
                        {c.email && (
                          <span className="flex items-center gap-1">
                            <Mail className="h-3 w-3" />
                            {c.email}
                          </span>
                        )}
                        {c.whatsapp && (
                          <span className="flex items-center gap-1">
                            <Phone className="h-3 w-3" />
                            {c.whatsapp}
                          </span>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>{c.visits_count}</TableCell>
                    <TableCell>
                      {c.last_visit
                        ? new Date(c.last_visit).toLocaleString()
                        : "—"}
                    </TableCell>
                    <TableCell>
                      ${c.average_ticket.toFixed(2)}
                    </TableCell>
                    <TableCell>
                      <div className="flex flex-wrap gap-1">
                        {c.preferences.length === 0 && (
                          <span className="text-xs text-muted-foreground">
                            Sin registro
                          </span>
                        )}
                        {c.preferences.map((p) => (
                          <Badge key={p} variant="outline" className="text-xs">
                            {p}
                          </Badge>
                        ))}
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex flex-wrap gap-1">
                        {c.allergies.length === 0 && (
                          <span className="text-xs text-muted-foreground">
                            Ninguna
                          </span>
                        )}
                        {c.allergies.map((a) => (
                          <Badge
                            key={a}
                            variant="destructive"
                            className="text-xs"
                          >
                            {a}
                          </Badge>
                        ))}
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
                {filtered.length === 0 && !isLoading && (
                  <TableRow>
                    <TableCell colSpan={7} className="text-center text-sm text-muted-foreground">
                      No se encontraron clientes con ese filtro.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
};

export default CRM;
