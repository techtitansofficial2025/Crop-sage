import React from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useLanguage, Language } from "@/contexts/LanguageContext";
import { Globe } from "lucide-react";

const LanguageSelector = () => {
  const { language, setLanguage, t } = useLanguage();

  const languages: { value: Language; label: string; nativeLabel: string }[] = [
    { value: 'en', label: t('english'), nativeLabel: 'English' },
    { value: 'hi', label: t('hindi'), nativeLabel: 'हिंदी' },
    { value: 'bn', label: t('bengali'), nativeLabel: 'বাংলা' },
    { value: 'ta', label: t('tamil'), nativeLabel: 'தமிழ்' },
    { value: 'te', label: t('telugu'), nativeLabel: 'తెలుగు' },
    { value: 'gu', label: t('gujarati'), nativeLabel: 'ગુજરાતી' },
    { value: 'mr', label: t('marathi'), nativeLabel: 'मराठी' },
    { value: 'pa', label: t('punjabi'), nativeLabel: 'ਪੰਜਾਬੀ' },
  ];

  return (
    <div className="flex items-center gap-2">
      <Globe className="w-4 h-4 text-muted-foreground" />
      <Select value={language} onValueChange={(value: Language) => setLanguage(value)}>
        <SelectTrigger className="w-40">
          <SelectValue placeholder={t('selectLanguage')} />
        </SelectTrigger>
        <SelectContent>
          {languages.map((lang) => (
            <SelectItem key={lang.value} value={lang.value}>
              <span className="flex items-center gap-2">
                <span>{lang.nativeLabel}</span>
              </span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
};

export default LanguageSelector;