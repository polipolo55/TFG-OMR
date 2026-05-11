% fig_staff_pitches.ly
% C major scale C4–C5 on the treble staff.
% Ghost voice at C5 (omit NoteHead+Stem) carries all pitch labels — every note
% at the same staff position → every label at exactly the same height above staff.
% "ledger line" annotation below C4 only.  LilyJAZZ engraving.

\version "2.26.0"
\include "lilyjazz.ily"
#(set-global-staff-size 24)
\header { tagline = ##f }

\paper {
  indent        = 0
  ragged-right  = ##t
  paper-width   = 92\mm
  top-margin    = 6\mm
  bottom-margin = 9\mm
  left-margin   = 6\mm
  right-margin  = 6\mm
  paper-height  = 62\mm
}

\score {
  \new Staff {
    <<
      % Visible C major scale C4–C5, stems down to keep clear of labels above
      \new Voice {
        \clef treble
        \key c \major
        \time 8/4
        \omit Staff.TimeSignature
        \stemDown
        \relative c' {
          c4_\markup { \sans \fontsize #-3 "ledger line" }
          d4 e4 f4 g4 a4 b4 c4
          \bar "|."
        }
      }
      % Ghost voice: all notes at C5 — identical position → identical label height
      \new Voice {
        \omit NoteHead
        \omit Stem
        \omit Flag
        \relative c'' {
          c4^\markup { \sans \bold \fontsize #-1 "C4" }
          c4^\markup { \sans \bold \fontsize #-1 "D4" }
          c4^\markup { \sans \bold \fontsize #-1 "E4" }
          c4^\markup { \sans \bold \fontsize #-1 "F4" }
          c4^\markup { \sans \bold \fontsize #-1 "G4" }
          c4^\markup { \sans \bold \fontsize #-1 "A4" }
          c4^\markup { \sans \bold \fontsize #-1 "B4" }
          c4^\markup { \sans \bold \fontsize #-1 "C5" }
        }
      }
    >>
  }
  \layout {
    \context { \Score \omit BarNumber }
  }
}
