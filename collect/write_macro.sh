#!/bin/bash
# This script writes out a Macro that can be used in SerialEM to collect a montage
# tilt-series. The user is prompted for the data collection directory, tile prefix,
# and tile coordinates file. The filenames of the acquired images follow the pattern:
# ${prefix}_${angle}_${tile_num}.mrc. The SerialEM macro is named acquire_montage.txt.
# Written by Songye Chen (songyech@caltech.edu).
#
# Usage: bash write_macro.sh [data directory] [image prefix] [tile coordinates file] 
#                        [no. tilt angles] [no. tiles per image] 
#
cat >> acquire_montage.txt <<EOF
MacroName MontageTiltSerieswithBeamCentering

##Realign to center image up to 3 times when the shift is more than 1um, end with nonzero image shift
##Center reference image taken at view mode and saved in buffer T
LmtRATimes = 3
LmtISShift = 1
ISDelay = 2

Require arrays
SetDirectory X:$1
OFPrefix = $2
ParFilename = $3
TSNumber = $4
TileNumber = $5

NewArray PlusStagePos -1 2
NewArray MinusStagePos -1 2
OpenTextFile MTSPar r 0 $ParFilename
NewArray Angle -1 1
NewArray MontIS_D -1 3
NewArray LastMontIS_D -1 3 

Loop $TSNumber
   Count = 0
   SetImageShift 0 0
   ReadLineToArray MTSPar Angle
   Echo Tilt to $Angle
   If $Angle == 0
      TiltTo $Angle
      ResetImageShift 
      Loop $LmtRATimes
         View
         AlignTo T
         ReportSpecimenShift 
         IS = $reportedValue1 * $reportedValue1 + $reportedValue2 * $reportedValue2
         If $IS > $LmtISShift
            ResetImageShift
         Else
            Break 
         Endif 
      EndLoop
      View
      Copy A R
      Copy A S
      ReportStageXYZ
      PlusStagePos[1] = $ReportedValue1
      PlusStagePos[2] = $ReportedValue2
      MinusStagePos = $PlusStagePos
      
      AutoFocus 
      Trial 
      CenterBeamFromImage 1

      ReadLineToArray MTSPar LastMontIS_D
      ImageShiftByMicrons $LastMontIS_D[1] $LastMontIS_D[2]
      output_file = $OFPrefix_$Angle_$Count.mrc
      OpenNewFile $output_file 
      Delay $ISDelay
      Record
      Save
      CloseFile 
      Echo $output_file saved
      CenterBeamFromImage 1
      Loop $TileNumber
         Count = $Count + 1
         ReadLineToArray MTSPar MontIS_D
         IS_X = $MontIS_D[1] - $LastMontIS_D[1]
         IS_Y = $MontIS_D[2] - $LastMontIS_D[2]
         ImageShiftByMicrons $IS_X $IS_Y
         ReportSpecimenShift
         output_file = $OFPrefix_$Angle_$Count.mrc
         OpenNewFile $output_file
         Delay $ISDelay
         Record
         Save
         CloseFile 
         Echo $output_file saved
         CenterBeamFromImage 1
         LastMontIS_D = $MontIS_D
      EndLoop
  
   ElseIf $Angle > 0
      TiltTo $Angle
      MoveStageTo $PlusStagePos[1] $PlusStagePos[2]
      Loop $LmtRATimes
         View
         AlignTo R
         ReportSpecimenShift 
         IS = $reportedValue1 * $reportedValue1 + $reportedValue2 * $reportedValue2
         If $IS > $LmtISShift
            ResetImageShift
         Else
            Break 
         Endif 
      EndLoop
      View
      Copy A R
      ReportStageXYZ
      PlusStagePos[1] = $ReportedValue1
      PlusStagePos[2] = $ReportedValue2
      
      AutoFocus
      Trial 
      CenterBeamFromImage 1

      ReadLineToArray MTSPar LastMontIS_D
      ImageShiftByMicrons $LastMontIS_D[1] $LastMontIS_D[2]
      if $LastMontIS_D[3] != 0
         ChangeFocus $LastMontIS_D[3]
         echo chang focus by $LastMontIS_D[3]
      endif
      output_file = $OFPrefix_$Angle_$Count.mrc
      OpenNewFile $output_file 
      Delay $ISDelay
      Record
      Save
      CloseFile 
      Echo $output_file saved
      CenterBeamFromImage 1
      Loop $TileNumber
         Count = $Count + 1
         ReadLineToArray MTSPar MontIS_D
         IS_X = $MontIS_D[1] - $LastMontIS_D[1]
         IS_Y = $MontIS_D[2] - $LastMontIS_D[2]
         DelDef = $MontIS_D[3] - $LastMontIS_D[3]
         ImageShiftByMicrons $IS_X $IS_Y
         if $DelDef != 0
            ChangeFocus $DelDef
            echo change focus by $DelDef
         Endif 
         ReportSpecimenShift
         output_file = $OFPrefix_$Angle_$Count.mrc
         OpenNewFile $output_file 
         Delay $ISDelay
         Record
         Save
         CloseFile 
         Echo $output_file saved
         CenterBeamFromImage 1
         LastMontIS_D = $MontIS_D
      EndLoop
  
   Else
      TiltTo ($Angle - 1)
      TiltTo $Angle
      MoveStageTo $MinusStagePos[1] $MinusStagePos[2]
      Loop $LmtRATimes
         View
         AlignTo S
         ReportSpecimenShift 
         IS = $reportedValue1 * $reportedValue1 + $reportedValue2 * $reportedValue2
         If $IS > $LmtISShift
            ResetImageShift
         Else
            Break 
         Endif 
      EndLoop
      View
      Copy A S
      ReportStageXYZ
      MinusStagePos[1] = $ReportedValue1
      MinusStagePos[2] = $ReportedValue2
      
      AutoFocus
      Trial 
      CenterBeamFromImage 1

      ReadLineToArray MTSPar LastMontIS_D
      ImageShiftByMicrons $LastMontIS_D[1] $LastMontIS_D[2]
      if $LastMontIS_D[3] != 0
         ChangeFocus $LastMontIS_D[3]
         echo change focus by $LastMontIS_D[3]
      Endif 
      output_file = $OFPrefix_$Angle_$Count.mrc
      OpenNewFile $output_file 
      Delay $ISDelay
      Record
      Save
      CloseFile 
      Echo $output_file saved
      CenterBeamFromImage 1
      Loop $TileNumber
         Count = $Count + 1
         ReadLineToArray MTSPar MontIS_D
         IS_X = $MontIS_D[1] - $LastMontIS_D[1]
         IS_Y = $MontIS_D[2] - $LastMontIS_D[2]
         DelDef = $MontIS_D[3] - $LastMontIS_D[3]
         if $DelDef != 0
            echo change focus by $DelDef
            ChangeFocus $DelDef
         Endif  
         ImageShiftByMicrons $IS_X $IS_Y
         ReportSpecimenShift
         output_file = $OFPrefix_$Angle_$Count.mrc
         OpenNewFile $output_file 
         Delay $ISDelay
         Record
         Save
         CloseFile 
         Echo $output_file saved
         CenterBeamFromImage 1
         LastMontIS_D = $MontIS_D
      EndLoop

   Endif 
EndLoop 

TiltTo 0
SetImageShift 0 0

EOF
