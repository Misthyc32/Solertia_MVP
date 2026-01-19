import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Bot, Loader2, Send, User } from "lucide-react";
import { toast } from "sonner";
import Analytics from "./Analytics";

const API_BASE_URL =
  import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const DASHBOARD_CUSTOMER_ID = "dashboard-demo";

type Message = {
  id: number;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
};

type BackendHistoryItem = {
  role: "user" | "assistant";
  content: string;
};

type ChatHistoryResponse = {
  customer_id: string;
  history: BackendHistoryItem[];
};

type ChatResponse = {
  response: string;
  route: string;
  pending_reservation: boolean;
  pending_update: boolean;
  pending_cancel: boolean;
  reservation_data?: Record<string, any> | null;
  customer_id: string;
};

// --- Subcomponent for the chat UI ---
const ChatContent = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const loadHistory = async () => {
      setIsLoading(true);
      try {
        const res = await fetch(
          `${API_BASE_URL}/chat/${encodeURIComponent(DASHBOARD_CUSTOMER_ID)}/history`,
        );

        if (!res.ok) {
          if (res.status !== 404) {
            const text = await res.text();
            console.error("Error al cargar historial:", text);
          }
          setMessages([
            {
              id: 1,
              role: "assistant",
              content:
                "¡Hola! Soy tu asistente de reservaciones. Pregúntame por horarios, disponibilidad o deja que te ayude a crear una reserva.",
              timestamp: new Date().toLocaleTimeString(),
            },
          ]);
          return;
        }

        const data: ChatHistoryResponse = await res.json();
        const mapped: Message[] = data.history.map((m, idx) => ({
          id: idx + 1,
          role: m.role,
          content: m.content,
          timestamp: new Date().toLocaleTimeString(),
        }));

        if (mapped.length === 0) {
          mapped.push({
            id: 1,
            role: "assistant",
            content:
              "¡Hola! Soy tu asistente de reservaciones. Pregúntame por horarios, disponibilidad o deja que te ayude a crear una reserva.",
            timestamp: new Date().toLocaleTimeString(),
          });
        }

        setMessages(mapped);
      } catch (err) {
        console.error(err);
        toast.error("No pude cargar el historial de chat.");
        setMessages([
          {
            id: 1,
            role: "assistant",
            content:
              "¡Hola! Soy tu asistente de reservaciones. Tu backend de chat parece no estar disponible, pero igual puedes escribir y revisamos.",
            timestamp: new Date().toLocaleTimeString(),
          },
        ]);
      } finally {
        setIsLoading(false);
      }
    };

    void loadHistory();
  }, []);

  const handleSendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: messages.length + 1,
      role: "user",
      content: input,
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const prompt = input;
    setInput("");
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: prompt,
          customer_id: DASHBOARD_CUSTOMER_ID,
        }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Error HTTP ${res.status}`);
      }

      const data: ChatResponse = await res.json();

      const assistantMessage: Message = {
        id: userMessage.id + 1,
        role: "assistant",
        content: data.response,
        timestamp: new Date().toLocaleTimeString(),
      };

      setMessages((prev) => [...prev, assistantMessage]);

      if (data.pending_reservation || data.pending_update || data.pending_cancel) {
        const info: string[] = [];
        if (data.pending_reservation)
          info.push("tengo una reservación pendiente de confirmar");
        if (data.pending_update)
          info.push("hay una actualización pendiente de reservación");
        if (data.pending_cancel)
          info.push("hay una cancelación pendiente de confirmar");

        const metaMessage: Message = {
          id: assistantMessage.id + 1,
          role: "assistant",
          content:
            "Nota interna: " +
            info.join(", ") +
            ". Completa el flujo desde el WhatsApp/UX correspondiente.",
          timestamp: new Date().toLocaleTimeString(),
        };

        setMessages((prev) => [...prev, metaMessage]);
      }
    } catch (err: any) {
      console.error(err);
      toast.error("Error al enviar el mensaje al backend.");
      const errorMessage: Message = {
        id: messages.length + 2,
        role: "assistant",
        content:
          "Hubo un problema al conectar con el servidor de chat. Intenta de nuevo en unos momentos.",
        timestamp: new Date().toLocaleTimeString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="h-[600px]">
      <CardHeader>
        <CardTitle>Asistente Conversacional</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col h-[calc(100%-60px)]">
        <ScrollArea className="flex-1 pr-4">
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${
                  message.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                {message.role === "assistant" && (
                  <div className="h-8 w-8 rounded-full bg-gradient-secondary flex items-center justify-center flex-shrink-0">
                    <Bot className="h-5 w-5 text-white" />
                  </div>
                )}
                {message.role === "user" && (
                  <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                    <User className="h-5 w-5 text-primary" />
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
                  <p className="text-xs opacity-70 mt-1">
                    {message.timestamp}
                  </p>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex gap-3 justify-start">
                <div className="h-8 w-8 rounded-full bg-gradient-secondary flex items-center justify-center flex-shrink-0">
                  <Loader2 className="h-5 w-5 text-white animate-spin" />
                </div>
                <div className="rounded-lg px-4 py-2 bg-muted">
                  <p className="text-sm">Pensando...</p>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>
        <div className="flex gap-2 mt-4">
          <Textarea
            placeholder="Escribe tu mensaje..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void handleSendMessage();
              }
            }}
            disabled={isLoading}
            className="min-h-[60px] max-h-[120px]"
          />
          <Button
            onClick={() => void handleSendMessage()}
            disabled={isLoading}
            className="self-end bg-gradient-primary"
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

// --- Main Page with toggle ---
const Chat = () => {
  const [activeTab, setActiveTab] = useState<"chat" | "analytics">("chat");

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-4xl font-bold tracking-tight">
          {activeTab === "chat" ? "Chat de Reservaciones" : "Manager Analytics"}
        </h1>
        <div className="flex gap-2">
          <Button
            variant={activeTab === "chat" ? "default" : "outline"}
            onClick={() => setActiveTab("chat")}
          >
            Chat
          </Button>
          <Button
            variant={activeTab === "analytics" ? "default" : "outline"}
            onClick={() => setActiveTab("analytics")}
          >
            Analytics
          </Button>
          <Button
            className="bg-green-500 hover:bg-green-600 text-white"
            onClick={() =>
              window.open(
                "http://wa.me/+14155238886?text=join%20identity-birth",
                "_blank",
              )
            }
          >
            Conectarse a WhatsApp
          </Button>
        </div>
      </div>

      {activeTab === "chat" ? <ChatContent /> : <Analytics />}
    </div>
  );
};

export default Chat;
