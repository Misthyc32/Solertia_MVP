import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Send, TrendingUp, DollarSign, MapPin } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { toast } from "sonner";

const API_BASE_URL =
  import.meta.env.VITE_API_URL ?? "http://localhost:8000";

type AnalyticsMessage = {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  hasChart?: boolean;
  chartUrl?: string;
};

type ManagerAskResponse = {
  text: string;
  plot_id?: string;
  plot_url?: string;
  error?: string;
};

const Analytics = () => {
  const [messages, setMessages] = useState<AnalyticsMessage[]>([
    {
      role: "assistant",
      content:
        "¡Hola! Soy tu asistente de analytics. Puedes preguntarme sobre revenue, SKUs, meseros y más. Ej: 'Top 10 SKUs por revenue para store_id=3'",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: AnalyticsMessage = {
      role: "user",
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const prompt = input;
    setInput("");
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE_URL}/manager/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Error HTTP ${res.status}`);
      }

      const data: ManagerAskResponse = await res.json();

      const assistantMessage: AnalyticsMessage = {
        role: "assistant",
        content: data.error
          ? `Error: ${data.error}`
          : data.text || "El agente no devolvió texto.",
        timestamp: new Date(),
        hasChart: Boolean(data.plot_url),
        chartUrl: data.plot_url,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err: any) {
      console.error(err);
      toast.error("Error al consultar el agente de analytics.");
      const errorMessage: AnalyticsMessage = {
        role: "assistant",
        content:
          "Hubo un problema al conectar con /manager/ask. Verifica que el backend esté levantado.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-4xl font-bold tracking-tight">
          Manager Analytics
        </h1>
        <p className="text-muted-foreground mt-2">
          AI-powered SQL analytics with natural language queries (conectado a /manager/ask)
        </p>
      </div>

      {/* Quick Stats dummy (puedes conectarlo luego a queries reales) */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Total Revenue
            </CardTitle>
            <DollarSign className="h-4 w-4 text-primary" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">$245,890</div>
            <p className="text-xs text-primary mt-1">+18% from last month</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Top SKU
            </CardTitle>
            <TrendingUp className="h-4 w-4 text-primary" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">Tacos al Pastor</div>
            <p className="text-xs text-muted-foreground mt-1">
              1,245 orders this week
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Store Locations
            </CardTitle>
            <MapPin className="h-4 w-4 text-primary" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">3</div>
            <p className="text-xs text-muted-foreground mt-1">
              Active locations
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Analytics Chat */}
      <Card className="h-[600px]">
        <CardHeader>
          <CardTitle>SQL Analytics Agent</CardTitle>
          <CardDescription>
            Ask questions in natural language to query the database via
            /manager/ask
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col h-[calc(100%-100px)]">
          <ScrollArea className="flex-1 pr-4">
            <div className="space-y-4">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex gap-3 ${
                    message.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  {message.role === "assistant" && (
                    <div className="h-8 w-8 rounded-full bg-gradient-secondary flex items-center justify-center flex-shrink-0">
                      <TrendingUp className="h-5 w-5 text-white" />
                    </div>
                  )}
                  <div
                    className={`rounded-lg px-4 py-2 max-w-[80%] ${
                      message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted"
                    }`}
                  >
                    <p className="text-sm whitespace-pre-line">
                      {message.content}
                    </p>
                    {message.hasChart && message.chartUrl && (
                      <div className="mt-3 p-2 bg-background rounded-lg border">
                        <img
                          src={message.chartUrl}
                          alt="Analytics chart"
                          className="w-full rounded-md"
                        />
                      </div>
                    )}
                    <p className="text-xs opacity-70 mt-1">
                      {message.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex gap-3 justify-start">
                  <div className="h-8 w-8 rounded-full bg-gradient-secondary flex items-center justify-center flex-shrink-0">
                    <TrendingUp className="h-5 w-5 text-white animate-pulse" />
                  </div>
                  <div className="rounded-lg px-4 py-2 bg-muted">
                    <p className="text-sm">Analizando...</p>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
          <div className="flex gap-2 mt-4">
            <Input
              placeholder="Ej: Top 10 SKUs por revenue para store_id=3"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !isLoading && handleSend()}
              disabled={isLoading}
            />
            <Button
              onClick={() => void handleSend()}
              disabled={isLoading}
              className="bg-gradient-secondary"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Store Map embebido desde /manager/ui */}
      <Card>
        <CardHeader>
          <CardTitle>Store Locations Map</CardTitle>
          <CardDescription>
            Interactive map served by your backend at /manager/ui
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] bg-muted rounded-lg overflow-hidden border">
            <iframe
              src={`${API_BASE_URL}/manager/ui`}
              title="Manager Analytics UI"
              className="w-full h-full border-0"
            />
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Analytics;