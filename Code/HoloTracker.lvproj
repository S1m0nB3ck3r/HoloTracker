<?xml version='1.0' encoding='UTF-8'?>
<Project Type="Project" LVVersion="23008000">
	<Property Name="NI.LV.All.SaveVersion" Type="Str">23.0</Property>
	<Property Name="NI.LV.All.SourceOnly" Type="Bool">true</Property>
	<Item Name="My Computer" Type="My Computer">
		<Property Name="server.app.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.control.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.tcp.enabled" Type="Bool">false</Property>
		<Property Name="server.tcp.port" Type="Int">0</Property>
		<Property Name="server.tcp.serviceName" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.tcp.serviceName.default" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.vi.callsEnabled" Type="Bool">true</Property>
		<Property Name="server.vi.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="specify.custom.address" Type="Bool">false</Property>
		<Item Name="subVis" Type="Folder">
			<Item Name="activateBtnIHM.vi" Type="VI" URL="../subVis/activateBtnIHM.vi"/>
			<Item Name="batchQueuesCommands.vi" Type="VI" URL="../subVis/batchQueuesCommands.vi"/>
			<Item Name="colorScatterRandomGenerator.vi" Type="VI" URL="../subVis/colorScatterRandomGenerator.vi"/>
			<Item Name="datas.ctl" Type="VI" URL="../subVis/datas.ctl"/>
			<Item Name="defValuesLink.ctl" Type="VI" URL="../subVis/defValuesLink.ctl"/>
			<Item Name="displayHoloListEnum.vi" Type="VI" URL="../subVis/displayHoloListEnum.vi"/>
			<Item Name="displayMessageSender.vi" Type="VI" URL="../subVis/displayMessageSender.vi"/>
			<Item Name="EcrireLireValeurParDefaut_HoloTrackPy_link.vi" Type="VI" URL="../subVis/EcrireLireValeurParDefaut_HoloTrackPy_link.vi"/>
			<Item Name="EcrireLireValeurParDefaut_HoloTrackPy_locate.vi" Type="VI" URL="../subVis/EcrireLireValeurParDefaut_HoloTrackPy_locate.vi"/>
			<Item Name="filterPortionTrajectories.vi" Type="VI" URL="../subVis/filterPortionTrajectories.vi"/>
			<Item Name="globState.vi" Type="VI" URL="../subVis/globState.vi"/>
			<Item Name="param_defaut.ctl" Type="VI" URL="../subVis/param_defaut.ctl"/>
			<Item Name="referencesBtns.ctl" Type="VI" URL="../subVis/referencesBtns.ctl"/>
			<Item Name="referencesLINKBtns.ctl" Type="VI" URL="../subVis/referencesLINKBtns.ctl"/>
			<Item Name="setScatterColor.vi" Type="VI" URL="../subVis/setScatterColor.vi"/>
		</Item>
		<Item Name="all_treatment_parameters.ctl" Type="VI" URL="../subVis/all_treatment_parameters.ctl"/>
		<Item Name="holoTracker.lvlib" Type="Library" URL="../HoloTrackerLib/holoTracker.lvlib"/>
		<Item Name="HoloTracker_link_labview_wrapper.py" Type="Document" URL="../HoloTrackerLib/HoloTracker_link_labview_wrapper.py"/>
		<Item Name="HoloTracker_Locate_Labview_wrapper.py" Type="Document" URL="../HoloTrackerLib/HoloTracker_Locate_Labview_wrapper.py"/>
		<Item Name="main_link.vi" Type="VI" URL="../main_link.vi"/>
		<Item Name="main_locate.vi" Type="VI" URL="../main_locate.vi"/>
		<Item Name="Dependencies" Type="Dependencies"/>
		<Item Name="Build Specifications" Type="Build">
			<Item Name="HoloTracker_Link" Type="EXE">
				<Property Name="App_copyErrors" Type="Bool">true</Property>
				<Property Name="App_INI_aliasGUID" Type="Str">{AB215352-5D9F-47CC-B559-678A6F82E43D}</Property>
				<Property Name="App_INI_GUID" Type="Str">{E0722688-E7C5-404C-9F3A-40D79DC502DB}</Property>
				<Property Name="App_serverConfig.httpPort" Type="Int">8002</Property>
				<Property Name="App_serverType" Type="Int">0</Property>
				<Property Name="Bld_autoIncrement" Type="Bool">true</Property>
				<Property Name="Bld_buildCacheID" Type="Str">{B463210B-6B22-4821-A5D4-73683C022327}</Property>
				<Property Name="Bld_buildSpecName" Type="Str">HoloTracker_Link</Property>
				<Property Name="Bld_excludeInlineSubVIs" Type="Bool">true</Property>
				<Property Name="Bld_excludeLibraryItems" Type="Bool">true</Property>
				<Property Name="Bld_excludePolymorphicVIs" Type="Bool">true</Property>
				<Property Name="Bld_localDestDir" Type="Path">../build/Holo_Tracker_Link</Property>
				<Property Name="Bld_localDestDirType" Type="Str">relativeToCommon</Property>
				<Property Name="Bld_modifyLibraryFile" Type="Bool">true</Property>
				<Property Name="Bld_previewCacheID" Type="Str">{37962870-B846-4A58-B4F6-207E2FD2E681}</Property>
				<Property Name="Bld_version.build" Type="Int">23</Property>
				<Property Name="Bld_version.major" Type="Int">1</Property>
				<Property Name="Destination[0].destName" Type="Str">HoloTracker_Link.exe</Property>
				<Property Name="Destination[0].path" Type="Path">../build/Holo_Tracker_Link/HoloTracker_Link.exe</Property>
				<Property Name="Destination[0].preserveHierarchy" Type="Bool">true</Property>
				<Property Name="Destination[0].type" Type="Str">App</Property>
				<Property Name="Destination[1].destName" Type="Str">Support Directory</Property>
				<Property Name="Destination[1].path" Type="Path">../build/Holo_Tracker_Link/data</Property>
				<Property Name="DestinationCount" Type="Int">2</Property>
				<Property Name="Source[0].itemID" Type="Str">{00166171-1EFC-423A-AD97-C6A399016A54}</Property>
				<Property Name="Source[0].type" Type="Str">Container</Property>
				<Property Name="Source[1].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[1].itemID" Type="Ref">/My Computer/main_link.vi</Property>
				<Property Name="Source[1].properties[0].type" Type="Str">Show menu bar</Property>
				<Property Name="Source[1].properties[0].value" Type="Bool">false</Property>
				<Property Name="Source[1].properties[1].type" Type="Str">Show vertical scroll bar</Property>
				<Property Name="Source[1].properties[1].value" Type="Bool">false</Property>
				<Property Name="Source[1].properties[2].type" Type="Str">Show horizontal scroll bar</Property>
				<Property Name="Source[1].properties[2].value" Type="Bool">false</Property>
				<Property Name="Source[1].properties[3].type" Type="Str">Show toolbar</Property>
				<Property Name="Source[1].properties[3].value" Type="Bool">false</Property>
				<Property Name="Source[1].properties[4].type" Type="Str">Show Abort button</Property>
				<Property Name="Source[1].properties[4].value" Type="Bool">false</Property>
				<Property Name="Source[1].propertiesCount" Type="Int">5</Property>
				<Property Name="Source[1].sourceInclusion" Type="Str">TopLevel</Property>
				<Property Name="Source[1].type" Type="Str">VI</Property>
				<Property Name="Source[10].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[10].itemID" Type="Ref">/</Property>
				<Property Name="Source[10].sourceInclusion" Type="Str">Include</Property>
				<Property Name="Source[11].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[11].itemID" Type="Ref">/</Property>
				<Property Name="Source[11].sourceInclusion" Type="Str">Include</Property>
				<Property Name="Source[2].destinationIndex" Type="Int">1</Property>
				<Property Name="Source[2].itemID" Type="Ref">/</Property>
				<Property Name="Source[2].sourceInclusion" Type="Str">Include</Property>
				<Property Name="Source[3].destinationIndex" Type="Int">1</Property>
				<Property Name="Source[3].itemID" Type="Ref">/</Property>
				<Property Name="Source[3].sourceInclusion" Type="Str">Include</Property>
				<Property Name="Source[4].destinationIndex" Type="Int">1</Property>
				<Property Name="Source[4].itemID" Type="Ref">/</Property>
				<Property Name="Source[4].sourceInclusion" Type="Str">Include</Property>
				<Property Name="Source[5].destinationIndex" Type="Int">1</Property>
				<Property Name="Source[5].itemID" Type="Ref">/</Property>
				<Property Name="Source[5].sourceInclusion" Type="Str">Include</Property>
				<Property Name="Source[6].destinationIndex" Type="Int">1</Property>
				<Property Name="Source[6].itemID" Type="Ref">/</Property>
				<Property Name="Source[6].sourceInclusion" Type="Str">Include</Property>
				<Property Name="Source[7].destinationIndex" Type="Int">1</Property>
				<Property Name="Source[7].itemID" Type="Ref">/</Property>
				<Property Name="Source[7].sourceInclusion" Type="Str">Include</Property>
				<Property Name="Source[8].destinationIndex" Type="Int">1</Property>
				<Property Name="Source[8].itemID" Type="Ref">/</Property>
				<Property Name="Source[8].sourceInclusion" Type="Str">Include</Property>
				<Property Name="Source[9].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[9].itemID" Type="Ref">/My Computer/main_locate.vi</Property>
				<Property Name="Source[9].type" Type="Str">VI</Property>
				<Property Name="SourceCount" Type="Int">12</Property>
				<Property Name="TgtF_companyName" Type="Str">LEMTA Université de Lorraine</Property>
				<Property Name="TgtF_fileDescription" Type="Str">HoloTracker_Link</Property>
				<Property Name="TgtF_internalName" Type="Str">HoloTracker_Link</Property>
				<Property Name="TgtF_legalCopyright" Type="Str">Copyright © 2023 LEMTA Université de Lorraine</Property>
				<Property Name="TgtF_productName" Type="Str">HoloTracker_Link</Property>
				<Property Name="TgtF_targetfileGUID" Type="Str">{BEB1D823-C86A-4559-A7CE-D2C475CE738D}</Property>
				<Property Name="TgtF_targetfileName" Type="Str">HoloTracker_Link.exe</Property>
				<Property Name="TgtF_versionIndependent" Type="Bool">true</Property>
			</Item>
			<Item Name="HoloTracker_Locate" Type="EXE">
				<Property Name="App_copyErrors" Type="Bool">true</Property>
				<Property Name="App_INI_aliasGUID" Type="Str">{82E47BCE-6B73-4CEA-AECE-D8B16381FE62}</Property>
				<Property Name="App_INI_GUID" Type="Str">{E04CFB57-8A2C-4F51-B62E-2CF618017C2B}</Property>
				<Property Name="App_serverConfig.httpPort" Type="Int">8002</Property>
				<Property Name="App_serverType" Type="Int">0</Property>
				<Property Name="Bld_autoIncrement" Type="Bool">true</Property>
				<Property Name="Bld_buildCacheID" Type="Str">{26BCDC13-CA62-4011-8852-89B24ED27FF6}</Property>
				<Property Name="Bld_buildSpecName" Type="Str">HoloTracker_Locate</Property>
				<Property Name="Bld_excludeInlineSubVIs" Type="Bool">true</Property>
				<Property Name="Bld_excludeLibraryItems" Type="Bool">true</Property>
				<Property Name="Bld_excludePolymorphicVIs" Type="Bool">true</Property>
				<Property Name="Bld_localDestDir" Type="Path">../build/HoloTracker_Locate</Property>
				<Property Name="Bld_localDestDirType" Type="Str">relativeToCommon</Property>
				<Property Name="Bld_modifyLibraryFile" Type="Bool">true</Property>
				<Property Name="Bld_previewCacheID" Type="Str">{D920B73B-F4AC-47EA-8DE0-671A5F4F9AAA}</Property>
				<Property Name="Bld_version.build" Type="Int">23</Property>
				<Property Name="Bld_version.major" Type="Int">1</Property>
				<Property Name="Destination[0].destName" Type="Str">HoloTracker_Locate.exe</Property>
				<Property Name="Destination[0].path" Type="Path">../build/HoloTracker_Locate/HoloTracker_Locate.exe</Property>
				<Property Name="Destination[0].preserveHierarchy" Type="Bool">true</Property>
				<Property Name="Destination[0].type" Type="Str">App</Property>
				<Property Name="Destination[1].destName" Type="Str">Support Directory</Property>
				<Property Name="Destination[1].path" Type="Path">../build/HoloTracker_Locate/Data</Property>
				<Property Name="DestinationCount" Type="Int">2</Property>
				<Property Name="Source[0].itemID" Type="Str">{A6D63D12-D8DD-42A6-98BA-F5927A8AA039}</Property>
				<Property Name="Source[0].type" Type="Str">Container</Property>
				<Property Name="Source[1].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[1].itemID" Type="Ref">/My Computer/main_locate.vi</Property>
				<Property Name="Source[1].properties[0].type" Type="Str">Show menu bar</Property>
				<Property Name="Source[1].properties[0].value" Type="Bool">false</Property>
				<Property Name="Source[1].properties[1].type" Type="Str">Show vertical scroll bar</Property>
				<Property Name="Source[1].properties[1].value" Type="Bool">false</Property>
				<Property Name="Source[1].properties[2].type" Type="Str">Show horizontal scroll bar</Property>
				<Property Name="Source[1].properties[2].value" Type="Bool">false</Property>
				<Property Name="Source[1].properties[3].type" Type="Str">Show toolbar</Property>
				<Property Name="Source[1].properties[3].value" Type="Bool">false</Property>
				<Property Name="Source[1].properties[4].type" Type="Str">Show Abort button</Property>
				<Property Name="Source[1].properties[4].value" Type="Bool">false</Property>
				<Property Name="Source[1].propertiesCount" Type="Int">5</Property>
				<Property Name="Source[1].sourceInclusion" Type="Str">TopLevel</Property>
				<Property Name="Source[1].type" Type="Str">VI</Property>
				<Property Name="SourceCount" Type="Int">2</Property>
				<Property Name="TgtF_companyName" Type="Str">LEMTA Université de Lorraine</Property>
				<Property Name="TgtF_fileDescription" Type="Str">HoloTracker_Locate</Property>
				<Property Name="TgtF_internalName" Type="Str">HoloTracker_Locate</Property>
				<Property Name="TgtF_legalCopyright" Type="Str">Copyright © 2023 LEMTA Université de Lorraine</Property>
				<Property Name="TgtF_productName" Type="Str">HoloTracker_Locate</Property>
				<Property Name="TgtF_targetfileGUID" Type="Str">{584F4DB5-D99A-410E-91A6-313BD823E65A}</Property>
				<Property Name="TgtF_targetfileName" Type="Str">HoloTracker_Locate.exe</Property>
				<Property Name="TgtF_versionIndependent" Type="Bool">true</Property>
			</Item>
			<Item Name="Installer_HoloTracker" Type="Installer">
				<Property Name="Destination[0].name" Type="Str">HoloTracker</Property>
				<Property Name="Destination[0].parent" Type="Str">{3912416A-D2E5-411B-AFEE-B63654D690C0}</Property>
				<Property Name="Destination[0].tag" Type="Str">{B9BC1E67-0CA5-4223-A675-215424C1F2D6}</Property>
				<Property Name="Destination[0].type" Type="Str">userFolder</Property>
				<Property Name="Destination[1].name" Type="Str">c:\HoloTracker</Property>
				<Property Name="Destination[1].path" Type="Path">/c/HoloTracker</Property>
				<Property Name="Destination[1].tag" Type="Str">{290588A0-698D-46B0-87F0-D8EC4253D378}</Property>
				<Property Name="Destination[1].type" Type="Str">absFolder</Property>
				<Property Name="DestinationCount" Type="Int">2</Property>
				<Property Name="DistPart[0].flavorID" Type="Str"></Property>
				<Property Name="DistPart[0].productID" Type="Str"></Property>
				<Property Name="DistPart[0].productName" Type="Str">NI LabVIEW Run-Time Engine 2023 (64-bit)</Property>
				<Property Name="DistPart[0].upgradeCode" Type="Str">{B5F88810-5FC9-3E79-B786-404C9235ADC9}</Property>
				<Property Name="DistPartCount" Type="Int">1</Property>
				<Property Name="INST_author" Type="Str">LEMTA Université de Lorraine</Property>
				<Property Name="INST_autoIncrement" Type="Bool">true</Property>
				<Property Name="INST_buildLocation" Type="Path">../build/Installer_HoloTracker</Property>
				<Property Name="INST_buildLocation.type" Type="Str">relativeToCommon</Property>
				<Property Name="INST_buildSpecName" Type="Str">Installer_HoloTracker</Property>
				<Property Name="INST_defaultDir" Type="Str">{B9BC1E67-0CA5-4223-A675-215424C1F2D6}</Property>
				<Property Name="INST_installerName" Type="Str">installHoloTracker.exe</Property>
				<Property Name="INST_productName" Type="Str">HoloTracker</Property>
				<Property Name="INST_productVersion" Type="Str">1.0.17</Property>
				<Property Name="InstSpecBitness" Type="Str">64-bit</Property>
				<Property Name="InstSpecVersion" Type="Str">23000000</Property>
				<Property Name="MSI_arpCompany" Type="Str">LEMTA Université de Lorraine</Property>
				<Property Name="MSI_autoselectDrivers" Type="Bool">true</Property>
				<Property Name="MSI_distID" Type="Str">{919827CA-27D9-4927-A981-5AD2138F6FB2}</Property>
				<Property Name="MSI_hideNonRuntimes" Type="Bool">true</Property>
				<Property Name="MSI_osCheck" Type="Int">0</Property>
				<Property Name="MSI_upgradeCode" Type="Str">{521CC0C9-0E45-401C-A393-70F7DCDEE517}</Property>
				<Property Name="RegDest[0].dirName" Type="Str">Software</Property>
				<Property Name="RegDest[0].dirTag" Type="Str">{DDFAFC8B-E728-4AC8-96DE-B920EBB97A86}</Property>
				<Property Name="RegDest[0].parentTag" Type="Str">2</Property>
				<Property Name="RegDestCount" Type="Int">1</Property>
				<Property Name="Source[0].dest" Type="Str">{290588A0-698D-46B0-87F0-D8EC4253D378}</Property>
				<Property Name="Source[0].File[0].dest" Type="Str">{290588A0-698D-46B0-87F0-D8EC4253D378}</Property>
				<Property Name="Source[0].File[0].name" Type="Str">HoloTracker_Link.exe</Property>
				<Property Name="Source[0].File[0].Shortcut[0].destIndex" Type="Int">0</Property>
				<Property Name="Source[0].File[0].Shortcut[0].name" Type="Str">HoloTrack_Link</Property>
				<Property Name="Source[0].File[0].Shortcut[0].subDir" Type="Str">HoloPyTracks</Property>
				<Property Name="Source[0].File[0].ShortcutCount" Type="Int">1</Property>
				<Property Name="Source[0].File[0].tag" Type="Str">{BEB1D823-C86A-4559-A7CE-D2C475CE738D}</Property>
				<Property Name="Source[0].FileCount" Type="Int">1</Property>
				<Property Name="Source[0].name" Type="Str">HoloTracker_Link</Property>
				<Property Name="Source[0].tag" Type="Ref">/My Computer/Build Specifications/HoloTracker_Link</Property>
				<Property Name="Source[0].type" Type="Str">EXE</Property>
				<Property Name="Source[1].dest" Type="Str">{290588A0-698D-46B0-87F0-D8EC4253D378}</Property>
				<Property Name="Source[1].File[0].dest" Type="Str">{290588A0-698D-46B0-87F0-D8EC4253D378}</Property>
				<Property Name="Source[1].File[0].name" Type="Str">HoloTracker_Locate.exe</Property>
				<Property Name="Source[1].File[0].Shortcut[0].destIndex" Type="Int">0</Property>
				<Property Name="Source[1].File[0].Shortcut[0].name" Type="Str">HoloTrack_Locate</Property>
				<Property Name="Source[1].File[0].Shortcut[0].subDir" Type="Str">HoloPyTracks</Property>
				<Property Name="Source[1].File[0].ShortcutCount" Type="Int">1</Property>
				<Property Name="Source[1].File[0].tag" Type="Str">{584F4DB5-D99A-410E-91A6-313BD823E65A}</Property>
				<Property Name="Source[1].FileCount" Type="Int">1</Property>
				<Property Name="Source[1].name" Type="Str">HoloTracker_Locate</Property>
				<Property Name="Source[1].tag" Type="Ref">/My Computer/Build Specifications/HoloTracker_Locate</Property>
				<Property Name="Source[1].type" Type="Str">EXE</Property>
				<Property Name="SourceCount" Type="Int">2</Property>
			</Item>
		</Item>
	</Item>
</Project>
