import React, { createContext, useContext, useState, ReactNode } from 'react';

export type Language = 'en' | 'hi' | 'bn' | 'ta' | 'te' | 'gu' | 'mr' | 'pa';

interface LanguageContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: string) => string;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export const useLanguage = () => {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
};

const translations = {
  en: {
    title: "Smart Crop Recommendations",
    subtitle: "AI-powered agricultural insights for optimal crop selection based on your soil and climate conditions",
    soilAnalysis: "Soil Analysis",
    climateOptimization: "Climate Optimization", 
    yieldPrediction: "Yield Prediction",
    fieldConditions: "Field Conditions",
    fieldConditionsDesc: "Enter your soil and environmental conditions for personalized crop recommendations",
    soilProperties: "Soil Properties",
    climateConditions: "Climate Conditions",
    soilNutrients: "Soil Nutrients (NPK)",
    soilType: "Soil Type",
    selectSoilType: "Select soil type",
    soilPh: "Soil pH",
    sowingMonth: "Sowing Month",
    selectMonth: "Select month",
    temperature: "Temperature (°C)",
    humidity: "Humidity (%)",
    nitrogen: "Nitrogen (N)",
    phosphorus: "Phosphorus (P)",
    potassium: "Potassium (K)",
    january: "January",
    february: "February", 
    march: "March",
    april: "April",
    may: "May",
    june: "June",
    july: "July",
    august: "August",
    september: "September",
    october: "October",
    november: "November",
    december: "December",
    getRecommendations: "Get Crop Recommendations",
    analyzingConditions: "Analyzing Conditions...",
    cropRecommendations: "Crop Recommendations",
    cropRecommendationsDesc: "AI-powered suggestions based on your specific conditions",
    enterConditions: "Enter your field conditions to receive personalized crop recommendations",
    waterSource: "Water Source",
    duration: "Duration",
    waterRequired: "Water Required",
    days: "days",
    variable: "Variable",
    featuresTitle: "Powered by Advanced AI Technology",
    featuresDesc: "Our machine learning model analyzes multiple factors to provide accurate crop recommendations",
    soilAnalysisDesc: "Comprehensive soil type and nutrient analysis for optimal crop selection",
    climateOptimizationDesc: "Weather pattern analysis to match crops with ideal growing conditions", 
    yieldPredictionDesc: "Predictive analytics for crop duration and water requirements",
    missingInfo: "Missing Information",
    fillAllFields: "Please fill in all fields to get accurate recommendations.",
    recommendationsGenerated: "Recommendations Generated",
    foundCrops: "Found {count} suitable crops for your conditions.",
    error: "Error",
    failedToGenerate: "Failed to generate recommendations. Please try again.",
    selectLanguage: "Select Language",
    english: "English",
    hindi: "हिंदी",
    bengali: "বাংলা", 
    tamil: "தமிழ்",
    telugu: "తెలుగు",
    gujarati: "ગુજરાતી",
    marathi: "मराठी",
    punjabi: "ਪੰਜਾਬੀ"
  },
  
  hi: {
    title: "स्मार्ट फसल सुझाव",
    subtitle: "आपकी मिट्टी और जलवायु की स्थिति के आधार पर इष्टतम फसल चयन के लिए AI-संचालित कृषि अंतर्दृष्टि",
    soilAnalysis: "मिट्टी विश्लेषण",
    climateOptimization: "जलवायु अनुकूलन",
    yieldPrediction: "उत्पादन पूर्वानुमान",
    fieldConditions: "खेत की स्थिति",
    fieldConditionsDesc: "व्यक्तिगत फसल सुझावों के लिए अपनी मिट्टी और पर्यावरणीय स्थितियां दर्ज करें",
    soilProperties: "मिट्टी के गुण",
    climateConditions: "जलवायु स्थितियां",
    soilNutrients: "मिट्टी के पोषक तत्व (NPK)",
    soilType: "मिट्टी का प्रकार",
    selectSoilType: "मिट्टी का प्रकार चुनें",
    soilPh: "मिट्टी का pH",
    sowingMonth: "बुआई का महीना",
    selectMonth: "महीना चुनें",
    temperature: "तापमान (°C)",
    humidity: "आर्द्रता (%)",
    nitrogen: "नाइट्रोजन (N)",
    phosphorus: "फास्फोरस (P)",
    potassium: "पोटेशियम (K)",
    january: "जनवरी",
    february: "फरवरी",
    march: "मार्च", 
    april: "अप्रैल",
    may: "मई",
    june: "जून",
    july: "जुलाई",
    august: "अगस्त",
    september: "सितंबर",
    october: "अक्टूबर",
    november: "नवंबर",
    december: "दिसंबर",
    getRecommendations: "फसल सुझाव प्राप्त करें",
    analyzingConditions: "स्थितियों का विश्लेषण...",
    cropRecommendations: "फसल सुझाव",
    cropRecommendationsDesc: "आपकी विशिष्ट स्थितियों के आधार पर AI-संचालित सुझाव",
    enterConditions: "व्यक्तिगत फसल सुझाव प्राप्त करने के लिए अपने खेत की स्थिति दर्ज करें",
    waterSource: "पानी का स्रोत",
    duration: "अवधि",
    waterRequired: "पानी की आवश्यकता",
    days: "दिन",
    variable: "परिवर्तनीय",
    featuresTitle: "उन्नत AI तकनीक द्वारा संचालित",
    featuresDesc: "हमारा मशीन लर्निंग मॉडल सटीक फसल सुझाव प्रदान करने के लिए कई कारकों का विश्लेषण करता है",
    soilAnalysisDesc: "इष्टतम फसल चयन के लिए व्यापक मिट्टी प्रकार और पोषक तत्व विश्लेषण",
    climateOptimizationDesc: "आदर्श बढ़ने की स्थितियों के साथ फसलों को मिलाने के लिए मौसम पैटर्न विश्लेषण",
    yieldPredictionDesc: "फसल की अवधि और पानी की आवश्यकताओं के लिए भविष्यसूचक विश्लेषण",
    missingInfo: "जानकारी गुम",
    fillAllFields: "सटीक सुझाव पाने के लिए कृपया सभी फ़ील्ड भरें।",
    recommendationsGenerated: "सुझाव तैयार किए गए",
    foundCrops: "आपकी स्थितियों के लिए {count} उपयुक्त फसलें मिलीं।",
    error: "त्रुटि",
    failedToGenerate: "सुझाव तैयार करने में विफल। कृपया पुनः प्रयास करें।",
    selectLanguage: "भाषा चुनें",
    english: "English",
    hindi: "हिंदी",
    bengali: "বাংলা",
    tamil: "தமிழ்",
    telugu: "తెలుగు", 
    gujarati: "ગુજરાતી",
    marathi: "मराठी",
    punjabi: "ਪੰਜਾਬੀ"
  }
};

interface LanguageProviderProps {
  children: ReactNode;
}

export const LanguageProvider: React.FC<LanguageProviderProps> = ({ children }) => {
  const [language, setLanguage] = useState<Language>('en');
  
  const t = (key: string): string => {
    const keys = key.split('.');
    let value: any = translations[language];
    
    for (const k of keys) {
      if (value && typeof value === 'object' && k in value) {
        value = value[k];
      } else {
        value = translations.en;
        for (const fallbackKey of keys) {
          if (value && typeof value === 'object' && fallbackKey in value) {
            value = value[fallbackKey];
          } else {
            return key;
          }
        }
        break;
      }
    }
    
    return typeof value === 'string' ? value : key;
  };

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      <div className={language === 'en' ? 'font-inter' : 
                     language === 'hi' || language === 'mr' ? 'font-hindi' :
                     language === 'bn' ? 'font-bengali' :
                     language === 'ta' ? 'font-tamil' :
                     language === 'te' ? 'font-telugu' :
                     language === 'gu' ? 'font-gujarati' : 'font-inter'}>
        {children}
      </div>
    </LanguageContext.Provider>
  );
};