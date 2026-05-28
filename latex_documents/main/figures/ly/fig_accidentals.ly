% fig_accidentals.ly
% The three accidentals applied to G4: a plain note, then sharp, flat, and natural.
% \cadenzaOn + barlines isolate each note so its accidental prints independently;
% g'!4 forces the natural sign to be typeset even though G is natural by default.
% autoBeaming off and \textLengthOn give each note its own labelled column.
% LilyJAZZ engraving, staff-size and label style matched to the other primer figures.

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
  paper-height  = 58\mm
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
    g'4   _\markup { \sans \fontsize #-1 \center-column { "G4" "(no accidental)" } } \bar "|"
    gis'4 _\markup { \sans \fontsize #-1 \center-column { "sharp" "+1 semitone" } }  \bar "|"
    ges'4 _\markup { \sans \fontsize #-1 \center-column { "flat" "-1 semitone" } }   \bar "|"
    g'!4  _\markup { \sans \fontsize #-1 \center-column { "natural" "cancels" } }    \bar "|."
  }
  \layout {
    \context { \Score \omit BarNumber }
  }
}
