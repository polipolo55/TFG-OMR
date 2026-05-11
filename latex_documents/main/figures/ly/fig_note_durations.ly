% fig_note_durations.ly
% Eight note values (whole → 32nd, plus dotted quarter and dotted eighth).
% All notes on C5 so _\markup labels all sit at the same distance below the staff —
% avoids the staggering that stems cause with ^markup above.
% \scaleDurations normalises every note to 1 quarter-note of visual space.
% \cadenzaOn disables bar counting.  Sans-serif labels.  LilyJAZZ engraving.

\version "2.26.0"
\include "lilyjazz.ily"
#(set-global-staff-size 20)
\header { tagline = ##f }

\paper {
  paper-width   = 200\mm
  top-margin    = 6\mm
  bottom-margin = 14\mm
  left-margin   = 10\mm
  right-margin  = 10\mm
  paper-height  = 50\mm
}

\score {
  \new Staff {
    \clef treble
    \omit Staff.TimeSignature
    \cadenzaOn
    \relative c'' {
      \stemUp
      \override TextScript.staff-padding = #1

      \scaleDurations 1/4 {
        c1_\markup { \sans \fontsize #-2 \center-align "whole" }
      }
      \bar "||"

      \scaleDurations 1/2 {
        c2_\markup { \sans \fontsize #-2 \center-align "half" }
      }
      \bar "||"

      c4_\markup { \sans \fontsize #-2 \center-align "quarter" }
      \bar "||"

      \scaleDurations 2/1 {
        c8_\markup { \sans \fontsize #-2 \center-align "eighth" }
      }
      \bar "||"

      \scaleDurations 4/1 {
        c16_\markup { \sans \fontsize #-2 \center-align "16th" }
      }
      \bar "||"

      \scaleDurations 8/1 {
        c32_\markup { \sans \fontsize #-2 \center-align "32nd" }
      }
      \bar "||"

      \scaleDurations 2/3 {
        c4._\markup { \sans \fontsize #-2 \center-align "dot. qtr" }
      }
      \bar "||"

      \scaleDurations 4/3 {
        c8._\markup { \sans \fontsize #-2 \center-align "dot. 8th" }
      }
      \bar "|."
    }
  }
  \layout {
    \context { \Score \omit BarNumber }
  }
}
