% fig_leadsheet_excerpt.ly
% A four-bar jazz lead sheet excerpt in C major (4/4), showing the canonical
% I–vi–ii–V chord progression (Cmaj7 – Am7 – Dm7 – G7) with a melody above.
% Rendered in LilyJAZZ style to match the project's training data aesthetic.

\version "2.26.0"
#(set-global-staff-size 18)
\include "lilyjazz.ily"
\header { tagline = ##f }
\paper {
  indent        = 10\mm
  ragged-right  = ##t
  top-margin    = 6\mm
  bottom-margin = 8\mm
  left-margin   = 8\mm
  right-margin  = 8\mm
  line-width    = 170\mm
}

harmonies = \chordmode {
  c1:maj7  a1:m7  d1:m7  g1:7
}

melody = \relative c'' {
  \clef treble
  \key c \major
  \time 4/4
  % Bar 1 — Cmaj7: G5 E5 D5 C5
  g'4 e d c |
  % Bar 2 — Am7: E5 C5 B4 A4
  e4 c b a |
  % Bar 3 — Dm7: D5 F5 E5 D5
  d4 f e d |
  % Bar 4 — G7: G4 B4 D5 G5
  g,4 b d g
  \bar "|."
}

\score {
  <<
    \new ChordNames {
      \set chordChanges = ##t
      \harmonies
    }
    \new Staff \melody
  >>
  \layout {
    \context { \Score \omit BarNumber }
  }
}
