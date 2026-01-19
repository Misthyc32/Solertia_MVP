import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Calendar, MessageSquare, TrendingUp, Users, UtensilsCrossed, BarChart3 } from "lucide-react";
import { Link } from "react-router-dom";

const stats = [
  {
    name: "Total Reservations",
    value: "284",
    change: "+12%",
    icon: Calendar,
    href: "/reservations",
  },
  {
    name: "Active Customers",
    value: "1,429",
    change: "+8%",
    icon: Users,
    href: "/crm",
  },
  {
    name: "Menu Items",
    value: "156",
    change: "+3%",
    icon: UtensilsCrossed,
    href: "/menu",
  },
  {
    name: "Weekly Revenue",
    value: "$24,580",
    change: "+15%",
    icon: TrendingUp,
    href: "/analytics",
  },
];

const quickActions = [
  {
    title: "Chat Assistant",
    description: "AI-powered restaurant assistant",
    icon: MessageSquare,
    href: "/chat",
    gradient: "bg-gradient-primary",
  },
  {
    title: "View Analytics",
    description: "Manager analytics dashboard",
    icon: BarChart3,
    href: "/analytics",
    gradient: "bg-gradient-secondary",
  },
  {
    title: "Manage Reservations",
    description: "View and manage bookings",
    icon: Calendar,
    href: "/reservations",
    gradient: "bg-gradient-primary",
  },
  {
    title: "Customer CRM",
    description: "Manage customer relationships",
    icon: Users,
    href: "/crm",
    gradient: "bg-gradient-secondary",
  },
];

const Dashboard = () => {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground mt-2">
          Welcome to your restaurant management system
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <Link key={stat.name} to={stat.href}>
            <Card className="hover:shadow-elegant transition-all cursor-pointer">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  {stat.name}
                </CardTitle>
                <stat.icon className="h-4 w-4 text-primary" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stat.value}</div>
                <p className="text-xs text-primary mt-1">{stat.change} from last week</p>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>

      {/* Quick Actions */}
      <div>
        <h2 className="text-2xl font-bold mb-4">Quick Actions</h2>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {quickActions.map((action) => (
            <Link key={action.title} to={action.href}>
              <Card className="hover:shadow-elegant transition-all cursor-pointer overflow-hidden h-full">
                <div className={`h-2 ${action.gradient}`} />
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-lg ${action.gradient}`}>
                      <action.icon className="h-5 w-5 text-white" />
                    </div>
                    <div>
                      <CardTitle className="text-base">{action.title}</CardTitle>
                      <CardDescription className="text-xs">
                        {action.description}
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
              </Card>
            </Link>
          ))}
        </div>
      </div>

      {/* Recent Activity */}
      <Card>
        <CardHeader>
          <CardTitle>System Status</CardTitle>
          <CardDescription>Backend API connection status</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Python Backend API</p>
                <p className="text-sm text-muted-foreground">
                  Configure your API endpoint in the environment
                </p>
              </div>
              <Button variant="outline" size="sm">
                Configure
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Dashboard;
