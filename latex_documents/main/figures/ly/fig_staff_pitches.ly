% fig_staff_pitches.ly
% C major scale C4–C5 on the treble staff with pitch labels above each note.
% Design notes:
%   - \textLengthOn gives each label its own horizontal column → no overlap and
%     no zigzag from collision avoidance.
%   - \override TextScript.staff-padding fixes the Y of all upper labels at one
%     height, regardless of notehead position.
%   - ragged-right = ##f spreads the eight notes across the full paper width.
%   - The "ledger line" annotation hangs well below the stems (\once padding)
%     so it never overlaps stems of D4–G4. LilyJAZZ engraving throughout.

\version "2.26.0"
\include "lilyjazz.ily"
#(set-global-staff-size 24)
\header { tagline = ##f }

\paper {
  indent        = 0
  ragged-right  = ##f
  paper-width   = 150\mm
  top-margin    = 8\mm
  bottom-margin = 16\mm
  left-margin   = 8\mm
  right-margin  = 8\mm
  paper-height  = 65\mm
}

\score {
  \new Staff \with {
    \omit TimeSignature
  } {
    \clef treble
    \key c \major
    \time 8/4
    \stemDown
    \textLengthOn
    \override TextScript.staff-padding = #2.5
    \override TextScript.outside-staff-priority = ##f
    \override Staff.LedgerLineSpanner.thickness = #1.6
    \override Staff.LedgerLineSpanner.length-fraction = #0.6
    \relative c' {
      % "ledger line" annotation: pushed below the stem area via staff-padding=7,
      % and \with-dimensions-from \null makes it zero-width so it does not widen
      % the C4 column.
      c4 ^\markup { \sans \bold "C4" }
         -\tweak staff-padding #7
         _\markup \with-dimensions-from \null
                  { \sans \italic \fontsize #-1.5 "ledger line" }
      d4 ^\markup { \sans \bold "D4" }
      e4 ^\markup { \sans \bold "E4" }
      f4 ^\markup { \sans \bold "F4" }
      g4 ^\markup { \sans \bold "G4" }
      a4 ^\markup { \sans \bold "A4" }
      b4 ^\markup { \sans \bold "B4" }
      c4 ^\markup { \sans \bold "C5" }
      \bar "|."
    }
  }
  \layout {
    \context { \Score \omit BarNumber }
  }
}
