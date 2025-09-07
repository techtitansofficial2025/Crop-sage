import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/hooks/use-toast";
import { Loader2, Leaf, Droplets, Calendar, TrendingUp, MapPin, Thermometer, Gauge } from "lucide-react";
import heroImage from "@/assets/agricultural-hero.jpg";

interface FormData {
  soil: string;
  sown: string;
  soilPh: string;
  temperature: string;
  humidity: string;
  nitrogen: string;
  phosphorus: string;
  potassium: string;
}

interface CropRecommendation {
  NAME: string;
  WATER_SOURCE: string | null;
  CROPDURATION: number;
  WATERREQUIRED: number;
  reason: string;
}

const Index = () => {
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [recommendations, setRecommendations] = useState<CropRecommendation[]>([]);
  const [formData, setFormData] = useState<FormData>({
    soil: '',
    sown: '',
    soilPh: '',
    temperature: '',
    humidity: '',
    nitrogen: '',
    phosphorus: '',
    potassium: ''
  });

  const soilTypes = [
    'Sandy', 'Clay', 'Silt', 'Loamy', 'Peaty', 'Chalky', 'Silty Clay', 'Sandy Loam', 'Clay Loam'
  ];

  const months = [
    { value: '1', label: 'January' },
    { value: '2', label: 'February' },
    { value: '3', label: 'March' },
    { value: '4', label: 'April' },
    { value: '5', label: 'May' },
    { value: '6', label: 'June' },
    { value: '7', label: 'July' },
    { value: '8', label: 'August' },
    { value: '9', label: 'September' },
    { value: '10', label: 'October' },
    { value: '11', label: 'November' },
    { value: '12', label: 'December' }
  ];

  const handleInputChange = (field: keyof FormData, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const simulateRecommendation = (): CropRecommendation[] => {
    // Simulate ML model recommendations based on inputs
    const mockRecommendations: CropRecommendation[] = [
      {
        NAME: "Rice",
        WATER_SOURCE: "Rainfed",
        CROPDURATION: 120,
        WATERREQUIRED: 1200,
        reason: "model"
      },
      {
        NAME: "Wheat",
        WATER_SOURCE: "Irrigated",
        CROPDURATION: 140,
        WATERREQUIRED: 800,
        reason: "model"
      },
      {
        NAME: "Maize",
        WATER_SOURCE: "Irrigated",
        CROPDURATION: 100,
        WATERREQUIRED: 600,
        reason: "model"
      },
      {
        NAME: "Cotton",
        WATER_SOURCE: "Irrigated",
        CROPDURATION: 180,
        WATERREQUIRED: 1000,
        reason: "model"
      },
      {
        NAME: "Soybean",
        WATER_SOURCE: "Rainfed",
        CROPDURATION: 110,
        WATERREQUIRED: 700,
        reason: "model"
      }
    ];

    return mockRecommendations;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Validate form
    const requiredFields = Object.entries(formData);
    const emptyFields = requiredFields.filter(([_, value]) => !value);
    
    if (emptyFields.length > 0) {
      toast({
        title: "Missing Information",
        description: "Please fill in all fields to get accurate recommendations.",
        variant: "destructive"
      });
      return;
    }

    setIsLoading(true);
    
    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const results = simulateRecommendation();
      setRecommendations(results);
      
      toast({
        title: "Recommendations Generated",
        description: `Found ${results.length} suitable crops for your conditions.`
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to generate recommendations. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <section className="relative h-[60vh] flex items-center justify-center overflow-hidden">
        <div 
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${heroImage})` }}
        />
        <div className="absolute inset-0 bg-gradient-primary opacity-75" />
        <div className="relative z-10 text-center px-4 max-w-4xl mx-auto">
          <h1 className="text-5xl md:text-6xl font-bold text-primary-foreground mb-4">
            Smart Crop Recommendations
          </h1>
          <p className="text-xl md:text-2xl text-primary-foreground/90 mb-8">
            AI-powered agricultural insights for optimal crop selection based on your soil and climate conditions
          </p>
          <div className="flex flex-wrap justify-center gap-4 text-primary-foreground/80">
            <div className="flex items-center gap-2">
              <Leaf className="w-5 h-5" />
              <span>Soil Analysis</span>
            </div>
            <div className="flex items-center gap-2">
              <Thermometer className="w-5 h-5" />
              <span>Climate Optimization</span>
            </div>
            <div className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              <span>Yield Prediction</span>
            </div>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <section className="py-12 px-4 max-w-7xl mx-auto">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Input Form */}
          <Card className="shadow-medium">
            <CardHeader className="bg-gradient-secondary">
              <CardTitle className="flex items-center gap-2">
                <MapPin className="w-5 h-5 text-primary" />
                Field Conditions
              </CardTitle>
              <CardDescription>
                Enter your soil and environmental conditions for personalized crop recommendations
              </CardDescription>
            </CardHeader>
            <CardContent className="p-6">
              <form onSubmit={handleSubmit} className="space-y-6">
                {/* Soil Information */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
                    <Leaf className="w-4 h-4 text-primary" />
                    Soil Properties
                  </h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="soil">Soil Type</Label>
                      <Select value={formData.soil} onValueChange={(value) => handleInputChange('soil', value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select soil type" />
                        </SelectTrigger>
                        <SelectContent>
                          {soilTypes.map(soil => (
                            <SelectItem key={soil} value={soil}>{soil}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="soilPh">Soil pH</Label>
                      <Input
                        id="soilPh"
                        type="number"
                        step="0.1"
                        min="0"
                        max="14"
                        placeholder="6.5"
                        value={formData.soilPh}
                        onChange={(e) => handleInputChange('soilPh', e.target.value)}
                      />
                    </div>
                  </div>
                </div>

                <Separator />

                {/* Climate Information */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
                    <Thermometer className="w-4 h-4 text-primary" />
                    Climate Conditions
                  </h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="sown">Sowing Month</Label>
                      <Select value={formData.sown} onValueChange={(value) => handleInputChange('sown', value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select month" />
                        </SelectTrigger>
                        <SelectContent>
                          {months.map(month => (
                            <SelectItem key={month.value} value={month.value}>{month.label}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="temperature">Temperature (°C)</Label>
                      <Input
                        id="temperature"
                        type="number"
                        min="-10"
                        max="50"
                        placeholder="25"
                        value={formData.temperature}
                        onChange={(e) => handleInputChange('temperature', e.target.value)}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="humidity">Humidity (%)</Label>
                      <Input
                        id="humidity"
                        type="number"
                        min="0"
                        max="100"
                        placeholder="65"
                        value={formData.humidity}
                        onChange={(e) => handleInputChange('humidity', e.target.value)}
                      />
                    </div>
                  </div>
                </div>

                <Separator />

                {/* NPK Values */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
                    <Gauge className="w-4 h-4 text-primary" />
                    Soil Nutrients (NPK)
                  </h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="nitrogen">Nitrogen (N)</Label>
                      <Input
                        id="nitrogen"
                        type="number"
                        min="0"
                        placeholder="40"
                        value={formData.nitrogen}
                        onChange={(e) => handleInputChange('nitrogen', e.target.value)}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="phosphorus">Phosphorus (P)</Label>
                      <Input
                        id="phosphorus"
                        type="number"
                        min="0"
                        placeholder="30"
                        value={formData.phosphorus}
                        onChange={(e) => handleInputChange('phosphorus', e.target.value)}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="potassium">Potassium (K)</Label>
                      <Input
                        id="potassium"
                        type="number"
                        min="0"
                        placeholder="35"
                        value={formData.potassium}
                        onChange={(e) => handleInputChange('potassium', e.target.value)}
                      />
                    </div>
                  </div>
                </div>

                <Button
                  type="submit"
                  className="w-full bg-gradient-primary hover:opacity-90 transition-all duration-300 text-lg py-6"
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                      Analyzing Conditions...
                    </>
                  ) : (
                    <>
                      <Leaf className="w-5 h-5 mr-2" />
                      Get Crop Recommendations
                    </>
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>

          {/* Recommendations */}
          <Card className="shadow-medium">
            <CardHeader className="bg-gradient-secondary">
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-primary" />
                Crop Recommendations
              </CardTitle>
              <CardDescription>
                AI-powered suggestions based on your specific conditions
              </CardDescription>
            </CardHeader>
            <CardContent className="p-6">
              {recommendations.length === 0 ? (
                <div className="text-center py-12">
                  <Leaf className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground">
                    Enter your field conditions to receive personalized crop recommendations
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {recommendations.map((crop, index) => (
                    <div 
                      key={index}
                      className="border border-border rounded-lg p-4 hover:shadow-soft transition-all duration-300"
                    >
                      <div className="flex justify-between items-start mb-3">
                        <h4 className="text-lg font-semibold text-foreground">{crop.NAME}</h4>
                        <Badge variant="secondary" className="bg-accent">
                          Rank #{index + 1}
                        </Badge>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div className="flex items-center gap-2">
                          <Droplets className="w-4 h-4 text-primary" />
                          <span className="text-muted-foreground">Water Source:</span>
                          <span className="font-medium">{crop.WATER_SOURCE || 'Variable'}</span>
                        </div>
                        
                        <div className="flex items-center gap-2">
                          <Calendar className="w-4 h-4 text-primary" />
                          <span className="text-muted-foreground">Duration:</span>
                          <span className="font-medium">{crop.CROPDURATION} days</span>
                        </div>
                      </div>
                      
                      <div className="mt-2 flex items-center gap-2 text-sm">
                        <Droplets className="w-4 h-4 text-primary" />
                        <span className="text-muted-foreground">Water Required:</span>
                        <span className="font-medium">{crop.WATERREQUIRED}mm</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-12 bg-gradient-secondary">
        <div className="max-w-6xl mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-foreground mb-4">
              Powered by Advanced AI Technology
            </h2>
            <p className="text-muted-foreground text-lg">
              Our machine learning model analyzes multiple factors to provide accurate crop recommendations
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-primary rounded-full flex items-center justify-center mx-auto mb-4">
                <Leaf className="w-8 h-8 text-primary-foreground" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Soil Analysis</h3>
              <p className="text-muted-foreground">
                Comprehensive soil type and nutrient analysis for optimal crop selection
              </p>
            </div>
            
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-primary rounded-full flex items-center justify-center mx-auto mb-4">
                <Thermometer className="w-8 h-8 text-primary-foreground" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Climate Optimization</h3>
              <p className="text-muted-foreground">
                Weather pattern analysis to match crops with ideal growing conditions
              </p>
            </div>
            
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-primary rounded-full flex items-center justify-center mx-auto mb-4">
                <TrendingUp className="w-8 h-8 text-primary-foreground" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Yield Prediction</h3>
              <p className="text-muted-foreground">
                Predictive analytics for crop duration and water requirements
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Index;