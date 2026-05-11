% fig_leadsheet_excerpt.ly
% Four-bar jazz lead sheet excerpt in C major (4/4).
% I–vi–ii–V chord progression: Cmaj7 – Am7 – Dm7 – G7.
% Melody mixes quarter and eighth notes; all pitches stay within the five-line staff
% (E4–D5), so no ledger lines appear above the staff.
% LilyJAZZ engraving style.

\version "2.26.0"
#(set-global-staff-size 18)
\include "lilyjazz.ily"
\header { tagline = ##f }

\paper {
  indent        = 10\mm
  ragged-right  = ##t
  paper-width   = 200\mm
  paper-height  = 42\mm
  top-margin    = 6\mm
  bottom-margin = 6\mm
  left-margin   = 15\mm
  right-margin  = 15\mm
}

harmonies = \chordmode {
  c1:maj7  a1:m7  d1:m7  g1:7
}

% Melody: all notes E4–D5, quarter + eighth mix.
% Bar 1 – Cmaj7: stepwise approach to C5 with rhythmic lift
% Bar 2 – Am7:   descend through chord tones back to E4
% Bar 3 – Dm7:   arch up to A4 then step back down
% Bar 4 – G7:    rise to D5 on a half-note landing
melody = \relative c' {
  \clef treble
  \key c \major
  \time 4/4
  e8 g a4 c b8 a |
  c4 b8 a g4 e |
  f4 a g8 f e4 |
  g4 b d2
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
