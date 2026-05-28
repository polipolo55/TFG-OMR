% fig_key_signatures.ly
% Four key signatures side by side: C major (none), G major (1 sharp),
% F major (1 flat), D major (2 sharps), each printed at the start of its measure.
% printKeyCancellation = ##f suppresses the courtesy naturals that would otherwise
% appear when one key changes to another, so each measure shows only its own signature.
% The anchor note is A4, which is diatonic in all four keys, so it never carries an
% accidental that would distract from the signature itself.
% LilyJAZZ engraving, matched to the other primer figures.

\version "2.26.0"
\include "lilyjazz.ily"
#(set-global-staff-size 24)
\header { tagline = ##f }

\paper {
  indent        = 0
  ragged-right  = ##f
  paper-width   = 160\mm
  top-margin    = 8\mm
  bottom-margin = 16\mm
  left-margin   = 8\mm
  right-margin  = 8\mm
  paper-height  = 58\mm
}

\score {
  \new Staff \with {
    \omit TimeSignature
  } {
    \clef treble
    \cadenzaOn
    \set Staff.printKeyCancellation = ##f
    \set Score.autoBeaming = ##f
    \textLengthOn
    \override TextScript.staff-padding = #2
    \override TextScript.outside-staff-priority = ##f
    \stemUp
    \key c \major a'1 _\markup { \sans \fontsize #-1 \center-column { "C major" "(none)" } }      \bar "||"
    \key g \major a'1 _\markup { \sans \fontsize #-1 \center-column { "G major" "1 sharp" } }     \bar "||"
    \key f \major a'1 _\markup { \sans \fontsize #-1 \center-column { "F major" "1 flat" } }      \bar "||"
    \key d \major a'1 _\markup { \sans \fontsize #-1 \center-column { "D major" "2 sharps" } }    \bar "|."
  }
  \layout {
    \context { \Score \omit BarNumber }
  }
}
