% fig_note_durations.ly
%
% Eight 4/4 bars, invisible (transparent) internal barlines + closing double bar only.
% Each bar is full width; symbols are *centred* on 16th / 32nd grids so the row reads
% evenly (no giant whitespace after short notes on beat 1).
%
% Pitch: \relative c'' and bare “c…”  C5 through the chain.
%

\version "2.26.0"
\include "lilyjazz.ily"

#(set-global-staff-size 22)

\header { tagline = ##f }

\paper {
  indent        = 0
  ragged-right  = ##t
  paper-width   = 560\mm
  paper-height  = 66\mm
  line-width    = 536\mm
  top-margin    = 10\mm
  bottom-margin = 44\mm
  left-margin   = 12\mm
  right-margin  = 12\mm
}

\score {
  \new Staff {
    \clef treble
    \time 4/4
    \omit Staff.TimeSignature

    \override Score.BarNumber.transparent = ##t
    \override Staff.BarLine.transparent = ##t

    \relative c'' {
      \override Score.BarNumber.transparent = ##t
      \stemUp

      %%
      %% Bar length = 16 sixteenths unless marked otherwise (bar 6 uses 32 thirty-seconds).
      %%

      %% 1 — whole (fills bar)
      c1_\markup { \raise #-2 \fontsize #-3 \sans "whole" }
      \bar "|"

      %% 2 — half centred: 2+4+2 eighths in the bar
      s8*2
      c2_\markup { \raise #-2 \fontsize #-3 \sans "half" }
      s8*2
      \bar "|"

      %% 3 — quarter
      s8*3
      c4_\markup { \raise #-2 \fontsize #-3 \sans "quarter" }
      s8*3
      \bar "|"

      %% 4 — eighth
      s16*7
      c8_\markup { \raise #-2 \fontsize #-3 \sans "eighth" }
      s16*7
      \bar "|"

      %% 5 — sixteenth
      s16*7
      c16_\markup { \raise #-2 \fontsize #-3 \sans "16th" }
      s16*8
      \bar "|"

      %% 6 — thirty-second (32 thirty-seconds per bar)
      s32*15
      c32_\markup { \raise #-2 \fontsize #-3 \sans "32nd" }
      s32*16
      \bar "|"

      %% 7 — dotted quarter (= 6 semiquavers)
      s16*5
      c4._\markup { \raise #-2 \fontsize #-3 \sans "dotted quarter" }
      s16*5
      \bar "|"

      %% 8 — dotted eighth (= 3 semiquavers)
      s16*6
      c8._\markup { \raise #-2 \fontsize #-3 \sans "dotted eighth" }
      s16*7

      \revert Staff.BarLine.transparent
      \bar "|."
    }
  }

  \layout {
    \context {
      \Score
      \omit BarNumber
    }
  }
}
