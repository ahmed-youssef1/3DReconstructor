import ROOT as R

# Function that plots TH1D histogram
def plot_histo(track_property):

    TracksAngCanvas = R.TCanvas("TracksAngCanvas", "Histogram of Angular Distribution of Tracks", 0, 0, 1000, 1000)
    TracksAngCanvas.cd()

    TracksAng = R.TH1D("TracksAng", "Angular Distribution of Tracks", 180, 0, 180)
    R.gStyle.SetPalette(55)

    for i in track_property:
        TracksAng.Fill(i)

    TracksAng.SetTitle("Angles of Tracks ; Angle[degree] ; Counts")
    TracksAng.Draw()

    R.gPad.Modified()
    R.gPad.Update()
    TracksAngCanvas.Update()

    TracksAngFile = R.TFile("TracksAngularDistribution.root", "RECREATE")
    TracksAngFile.cd()

    TracksAngCanvas.Write()
    TracksAngFile.Close()