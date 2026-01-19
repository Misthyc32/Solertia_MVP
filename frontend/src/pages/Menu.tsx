import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Search, Plus, UtensilsCrossed } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";

const API_BASE_URL =
  import.meta.env.VITE_API_URL ?? "http://localhost:8000";

type MenuItem = {
  id: string;
  name: string;
  description: string;
  price: number;
  category: string;
  available: boolean;
};

type MenuSearchResult = {
  content: string;
  category?: string;
  metadata?: Record<string, any>;
};

const Menu = () => {
  const [menuItems, setMenuItems] = useState<MenuItem[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("Todos");
  const [categories, setCategories] = useState<string[]>(["Todos"]);
  const [isLoading, setIsLoading] = useState(false);

  const mapResultToItem = (result: MenuSearchResult, index: number): MenuItem => {
    const meta = result.metadata || {};
    const name =
      meta.name || meta.nombre || result.content || "Sin nombre";
    const description =
      meta.description || meta.descripcion || result.content || "";
    const priceRaw = meta.price ?? meta.precio ?? 0;
    const price = typeof priceRaw === "number" ? priceRaw : parseFloat(priceRaw);
    const category =
      result.category || meta.category || meta.categoria || "Sin categoría";

    return {
      id: String(meta.sku || index),
      name,
      description,
      price: Number.isNaN(price) ? 0 : price,
      category,
      available: meta.is_active ?? true,
    };
  };

  const fetchCategories = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/menu/categories`);
      if (!res.ok) return;
      const data: { categories: string[] } = await res.json();
      if (Array.isArray(data.categories) && data.categories.length > 0) {
        setCategories(["Todos", ...data.categories]);
      }
    } catch (err) {
      console.error("Error loading categories", err);
    }
  };

  const fetchMenu = async () => {
    setIsLoading(true);
    try {
      const query =
        searchQuery || (selectedCategory !== "Todos" ? selectedCategory : "");

      const res = await fetch(`${API_BASE_URL}/menu/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query,
          limit: 50,
        }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Error HTTP ${res.status}`);
      }

      const data: { query: string; results: MenuSearchResult[] } =
        await res.json();

      const mapped = (data.results || []).map(mapResultToItem);
      setMenuItems(mapped);
    } catch (err: any) {
      console.error(err);
      toast.error("Error al buscar en el menú.");
      setMenuItems([]);
    } finally {
      setIsLoading(false);
    }
  };

  // Cargar categorías y menú inicial
  useEffect(() => {
    void fetchCategories();
    void fetchMenu();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Volver a buscar cuando cambie categoría o texto
  useEffect(() => {
    void fetchMenu();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchQuery, selectedCategory]);

  const filteredItems = menuItems.filter((item) => {
    const matchesSearch =
      item.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.description.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesCategory =
      selectedCategory === "Todos" || item.category === selectedCategory;

    return matchesSearch && matchesCategory;
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold tracking-tight">Menu Management</h1>
          <p className="text-muted-foreground mt-2">
            Connected to your Python menu API (/menu/search y /menu/categories)
          </p>
        </div>
        <Button className="bg-gradient-primary" disabled>
          <Plus className="mr-2 h-4 w-4" />
          Add Item
        </Button>
      </div>

      {/* Search and Filter */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col gap-4 md:flex-row md:items-center">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Search menu items..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
            <Tabs
              value={selectedCategory}
              onValueChange={setSelectedCategory}
              className="w-full md:w-auto"
            >
              <TabsList>
                {categories.map((category) => (
                  <TabsTrigger key={category} value={category} className="text-xs">
                    {category}
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>
          </div>
        </CardContent>
      </Card>

      {/* Menu Items Grid */}
      <Tabs value="grid">
        <TabsContent value="grid">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {filteredItems.map((item) => (
              <Card
                key={item.id}
                className="hover:shadow-elegant transition-all"
              >
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-2">
                      <div className="h-10 w-10 rounded-lg bg-gradient-primary flex items-center justify-center">
                        <UtensilsCrossed className="h-5 w-5 text-white" />
                      </div>
                      <div>
                        <CardTitle className="text-base">{item.name}</CardTitle>
                        <Badge variant="outline" className="mt-1 text-xs">
                          {item.category}
                        </Badge>
                      </div>
                    </div>
                    <Badge
                      variant={item.available ? "default" : "secondary"}
                      className={item.available ? "bg-primary" : ""}
                    >
                      {item.available ? "Available" : "Unavailable"}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <CardDescription className="mb-4">
                    {item.description}
                  </CardDescription>
                  <div className="flex items-center justify-between">
                    <span className="text-2xl font-bold text-primary">
                      ${item.price.toFixed(2)}
                    </span>
                    <Button variant="outline" size="sm" disabled>
                      Edit
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {filteredItems.length === 0 && !isLoading && (
            <Card>
              <CardContent className="py-12 text-center">
                <UtensilsCrossed className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">
                  No menu items found
                </h3>
                <p className="text-sm text-muted-foreground">
                  Try adjusting your search or filters
                </p>
              </CardContent>
            </Card>
          )}

          {isLoading && (
            <p className="text-sm text-muted-foreground">Loading menu...</p>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Menu;