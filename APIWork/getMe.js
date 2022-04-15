const fs = require('fs')
const SpotifyWebApi = require('spotify-web-api-node');
const token = "BQBUcWPZKZYoAOFwER_bBM-h6afpOGlZVmQeo3QeKAmG2Yu2p5Ma3Uq2pMHXdn1a92ueBPaXJsbLS6WVeK_sDs67G4d04ePzxBRgc95rpXMRA6uUMTsCsI_uBJoMXjOqNVDvGZ2fDLguzkw0x-MxAY_pySJ0V-sZgGCdPQpGRhd1iqdKdzYL75B5f0fUaFGut1len7InYSpvoBe9inLjBTVFQlGKv66BI28imTUfiLNsSM0SkCPD7vHT_PBnu6e1P2Ds2kFwweyGSp8rvUzhAZtkYKxJUU9XaEbKlnVRzsPUGTl1pEam";

const spotifyApi = new SpotifyWebApi();
spotifyApi.setAccessToken(token);

//GET MY PROFILE DATA
function getMyData() {
  (async () => {
    const me = await spotifyApi.getMe();
    // console.log(me.body);
    // getUserPlaylists(me.body.id);
    getSavedTracks()
  })().catch(e => {
    console.error(e);
  });
}

async function getSavedTracks() {
  headers = "title,artist,id,danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo\n"
  data = []
  var temp = ""
  // for (var i = 0; i < currTrack.length; ++i) {
  //   temp += currTrack[i] + ", \n"
  // }
  temp += headers
  offsetVar = 0
  do {
    data = await spotifyApi.getMySavedTracks({
      offset: offsetVar,
      limit: 50
    })
    // console.log(data.body.items.length)

    for (let track_obj of data.body.items) {
      let currTrack = []

      const track = track_obj.track
      trackFeatures = await getTrackAudioFeature(track.id)
      bod = trackFeatures.body

      // const tracksJSON = { bod }
      // let trackData = JSON.stringify(bod);
      // bod = JSON.parse(trackData)

      trackInfo = []
      trackInfo.push("\"" + track.name + "\"")
      trackInfo.push("\"" + track.artists[0].name + "\"")
      trackInfo.push("\"" + track.id + "\"")
      for (const [key, value] of Object.entries(bod)){
        // console.log(key)
        if (key != "id" && key != "track_href" && key != "type" && key != "uri" && key != "analysis_url") {
          trackInfo.push(value)
        }
      }

      currTrack.push(trackInfo);
      // console.log(track.name + " : " + track.artists[0].name + " : " + track.id + " : " + trackData)

      for (var i = 0; i < currTrack.length; ++i) {
        temp += currTrack[i] + "\n"
      }

      // console.log("TEMP " + temp)
      

    }
    offsetVar += 50
  } while (data.body.items.length == 50);
  fs.writeFileSync("likedSongs" + '.csv', temp);
}

//GET MY PLAYLISTS
async function getUserPlaylists(userName) {
  const data = await spotifyApi.getUserPlaylists(userName)

  console.log("---------------+++++++++++++++++++++++++")
  let playlists = []

  var count = 0
  for (let playlist of data.body.items) {
    if (count == 0) {
      console.log(playlist.name + " " + playlist.id)
      
      let tracks = await getPlaylistTracks(playlist.id, playlist.name);
      // console.log(tracks);

      // const tracksJSON = { tracks }
      // let data = JSON.stringify(tracksJSON);
      // fs.writeFileSync(playlist.name+'.json', data);
    }
  }
}

//GET SONGS FROM PLAYLIST
async function getPlaylistTracks(playlistId, playlistName) {

  const data = await spotifyApi.getPlaylistTracks(playlistId, {
    offset: 1,
    limit: 5,
    fields: 'items'
  })

  // console.log('The playlist contains these tracks', data.body);
  // console.log('The playlist contains these tracks: ', data.body.items[0].track);
  // console.log("'" + playlistName + "'" + ' contains these tracks:');
  let tracks = [];

  for (let track_obj of data.body.items) {
    const track = track_obj.track
    trackFeatures = await getTrackAudioFeature(track.id)
    tracks.push([track.name, track.artists[0].name, track.id, trackFeatures.body]);
    // console.log(track.name + " " + track.artists[0].name + " " + track.id)
  }

  const tracksJSON = { tracks }
  let trackData = JSON.stringify(tracksJSON);
  fs.writeFileSync("playLists/" + encodeURIComponent(playlistName)+'.json', trackData);

  console.log("---------------+++++++++++++++++++++++++")
  return tracks;
}

// GET AUDIO FEATURES FOR A SONG
async function getTrackAudioFeature(trackId) {

  const data = await spotifyApi.getAudioFeaturesForTrack(trackId)

  // console.log(data.body)
  return data
}

getMyData();
