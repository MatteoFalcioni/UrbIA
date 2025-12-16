/**
 * ScoreBar: Visual indicator for analysis quality score from reviewer
 * Shows a colored progress bar based on the score value
 */

interface ScoreBarProps {
  score: number | null;  // Score from 0 to 1, or null if not available
}

export function ScoreBar({ score }: ScoreBarProps) {
  // Don't render if no score available
  if (score === null || score === undefined) {
    return null;
  }

  // Clamp score between 0 and 1
  const clampedScore = Math.max(0, Math.min(1, score));
  const percentage = Math.round(clampedScore * 100);

  // Determine color based on score thresholds
  let colorClasses = '';
  let bgColorClasses = '';
  
  if (clampedScore > 0.7) {
    // Green for high scores
    colorClasses = 'bg-green-500 dark:bg-green-500';
    bgColorClasses = 'bg-green-100 dark:bg-green-900/30';
  } else if (clampedScore >= 0.4) {
    // Yellow for medium scores
    colorClasses = 'bg-yellow-500 dark:bg-yellow-500';
    bgColorClasses = 'bg-yellow-100 dark:bg-yellow-900/30';
  } else {
    // Orange for low scores
    colorClasses = 'bg-orange-500 dark:bg-orange-500';
    bgColorClasses = 'bg-orange-100 dark:bg-orange-900/30';
  }

  return (
    <div className="flex items-center gap-2 px-3 py-1">
      <span className="text-xs font-medium text-gray-600 dark:text-slate-400 whitespace-nowrap">
        Analysis Quality
      </span>
      <div className={`flex-1 h-2 rounded-full overflow-hidden ${bgColorClasses}`}>
        <div
          className={`h-full transition-all duration-500 ease-out ${colorClasses}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="text-xs font-semibold text-gray-700 dark:text-slate-300 min-w-[2.5rem] text-right">
        {percentage}%
      </span>
    </div>
  );
}
