% fig_note_durations.ly
% Eight note values on C5: whole, half, quarter, eighth, 16th, 32nd, dotted quarter,
% dotted eighth.
% \cadenzaOn disables bar counting so heterogeneous durations sit on one line.
% \textLengthOn makes each note's column at least as wide as its label below;
% combined with uniform label sizing this yields roughly equal horizontal slots,
% which avoids the squashed whole note produced by \scaleDurations.
% autoBeaming is disabled so the short notes (8, 16, 32) keep their flags
% instead of being beamed together. LilyJAZZ engraving.

\version "2.26.0"
\include "lilyjazz.ily"
#(set-global-staff-size 24)
\header { tagline = ##f }

% Aspect ratio (W/H ≈ 2.4) and staff-size matched to fig_staff_pitches.ly so
% that, when both figures are scaled to width = 0.82\textwidth in the thesis,
% their displayed heights match — see \TFGmusicStaffDispHeight in main.tex.
\paper {
  indent        = 0
  ragged-right  = ##f
  paper-width   = 180\mm
  top-margin    = 18\mm
  bottom-margin = 24\mm
  left-margin   = 10\mm
  right-margin  = 10\mm
  paper-height  = 78\mm
}

\score {
  \new Staff \with {
    \omit TimeSignature
  } {
    \clef treble
    \cadenzaOn
    \set Score.autoBeaming = ##f
    \textLengthOn
    \override TextScript.staff-padding = #2
    \override TextScript.outside-staff-priority = ##f
    \stemUp
    \relative c'' {
      c1  _\markup { \sans \fontsize #-1 \center-align "whole" }
      \bar "|"
      c2  _\markup { \sans \fontsize #-1 \center-align "half" }
      \bar "|"
      c4  _\markup { \sans \fontsize #-1 \center-align "quarter" }
      \bar "|"
      c8  _\markup { \sans \fontsize #-1 \center-align "eighth" }
      \bar "|"
      c16 _\markup { \sans \fontsize #-1 \center-align "16th" }
      \bar "|"
      c32 _\markup { \sans \fontsize #-1 \center-align "32nd" }
      \bar "|"
      c4. _\markup { \sans \fontsize #-1 \center-align "dotted quarter" }
      \bar "|"
      c8. _\markup { \sans \fontsize #-1 \center-align "dotted eighth" }
      \bar "|."
    }
  }
  \layout {
    \context { \Score \omit BarNumber }
  }
}
