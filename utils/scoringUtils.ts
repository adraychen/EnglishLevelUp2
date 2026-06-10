// Number word mappings for normalization
const numberWords: Record<string, string> = {
  '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
  '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
  '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
  '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
  '18': 'eighteen', '19': 'nineteen', '20': 'twenty', '30': 'thirty',
  '40': 'forty', '50': 'fifty', '60': 'sixty', '70': 'seventy',
  '80': 'eighty', '90': 'ninety', '100': 'hundred', '1000': 'thousand'
};

// Convert digits to words
const convertNumbersToWords = (text: string): string => {
  return text.replace(/\b\d+\b/g, (match) => {
    if (numberWords[match]) {
      return numberWords[match];
    }
    const num = parseInt(match, 10);
    if (num >= 21 && num <= 99) {
      const tens = Math.floor(num / 10) * 10;
      const ones = num % 10;
      if (ones === 0) {
        return numberWords[tens.toString()] || match;
      }
      const tensWord = numberWords[tens.toString()];
      const onesWord = numberWords[ones.toString()];
      if (tensWord && onesWord) {
        return `${tensWord} ${onesWord}`;
      }
    }
    return match;
  });
};

export const normalize = (text: string): string => {
  let result = text
    .toLowerCase()
    .replace(/[\u2018\u2019\u2032]/g, "'")
    .replace(/[^a-z0-9\s']/g, "")
    .replace(/\s+/g, " ")
    .trim();
  result = convertNumbersToWords(result);
  return result;
};

export interface WordResult {
  word: string;
  status: 'hit' | 'miss';
}

export interface ScoreResult {
  label: string;
  accuracy: number;
  color: string;
  bg: string;
  passed: boolean;
  wordResults: WordResult[];
}

export const scorePronunciation = (target: string, spoken: string): ScoreResult | null => {
  if (!spoken) return null;

  const targetNorm = normalize(target);
  const spokenNorm = normalize(spoken);

  const targetWords = targetNorm.split(" ");
  const spokenPool = [...spokenNorm.split(" ")];

  let matches = 0;

  const wordResults: WordResult[] = target.split(" ").map((originalWord) => {
    const cleanWord = normalize(originalWord);
    const index = spokenPool.indexOf(cleanWord);

    if (index !== -1) {
      matches++;
      spokenPool.splice(index, 1);
      return { word: originalWord, status: 'hit' };
    } else {
      return { word: originalWord, status: 'miss' };
    }
  });

  const baseAccuracy = matches / targetWords.length;
  const extraWords = spokenPool.length;
  const extraWordPenalty = extraWords * 0.05;
  const accuracy = Math.max(0, Math.round((baseAccuracy - extraWordPenalty) * 100));

  if (accuracy >= 90) {
    return {
      label: "Excellent!",
      accuracy,
      color: "text-green-700",
      bg: "bg-green-100",
      passed: true,
      wordResults
    };
  }

  if (accuracy >= 70) {
    return {
      label: "Almost there",
      accuracy,
      color: "text-orange-700",
      bg: "bg-orange-100",
      passed: false,
      wordResults
    };
  }

  return {
    label: "Try again",
    accuracy,
    color: "text-red-700",
    bg: "bg-red-100",
    passed: false,
    wordResults
  };
};
