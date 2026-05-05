% fig_staff_pitches.ly
% C major scale C4–C5; one 8/4 bar → even quarter spacing (no stray barline mid-scale).
% LilyJAZZ. Small paper-height + no fixed line-width → one horizontal system (see lilypond_render).
% C5: after B4 write c4 in \relative c' — not c'.

\version "2.26.0"
\include "lilyjazz.ily"
#(set-global-staff-size 22)
\header { tagline = ##f }

\paper {
  indent        = 0
  ragged-right  = ##t
  top-margin    = 6\mm
  bottom-margin = 8\mm
  left-margin   = 6\mm
  right-margin  = 6\mm
  paper-height  = 54\mm
}

\score {
  \new Staff {
    \clef treble
    \key c \major
    \time 8/4
    \omit Staff.TimeSignature
    \relative c' {
      \override Score.BarNumber.transparent = ##t
      \override Stem.direction = #UP
      c4_\markup { \fontsize #-2 \center-align "C4" }
      d4_\markup { \fontsize #-2 \center-align "D4" }
      e4_\markup { \fontsize #-2 \center-align "E4" }
      f4_\markup { \fontsize #-2 \center-align "F4" }
      g4_\markup { \fontsize #-2 \center-align "G4" }
      a4_\markup { \fontsize #-2 \center-align "A4" }
      b4_\markup { \fontsize #-2 \center-align "B4" }
      c4_\markup { \fontsize #-2 \center-align "C5" }
      \bar "|."
    }
  }
  \layout {
    \context { \Score \omit BarNumber }
  }
}
