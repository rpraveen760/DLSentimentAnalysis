��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
=
Greater
x"T
y"T
z
"
Ttype:
2	
�
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype�
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype�
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype�
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype�
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
�
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint���������"serve*2.9.22v2.9.1-132-g18960c44ad38��
|
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_7*
value_dtype0	
k

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name569*
value_dtype0	
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
��
Const_4Const*
_output_shapes	
:�N*
dtype0*��
value��B���NBtheBaBinBtoBofBandBiBisBforBonByouBmyBwithBitBthatBatBbyBthisBfromBbeBareBwasBhaveBlikeBasBupBsoBjustBbutBmeBimByourBnotBampBoutBitsBwillBanBnoBhasBfireBafterBallBwhenBweBifBnowBviaBnewBmoreBgetBorBaboutBwhatBheBpeopleBnewsBbeenBoverBoneBhowBdontBtheyBwhoBintoBwereBdoBusB2BcanBvideoB	emergencyBthereBdisasterBthanBpoliceBwouldBhisBstillBherBsomeBbodyBstormBcrashBburningBsuicideBbackBmanB
californiaBwhyBtimeBthemBhadB	buildingsBrtBfirstBcantBseeBgotBdayBoffBourBgoingBnuclearBknowBworldBbombBfiresBloveBkilledBgoBattackByoutubeBdeadBtwoBfamiliesB3BtrainBfullBbeingBwarBmanyBtodayBthinkBonlyBcarBaccidentBlifeB	hiroshimaBtheirBsayBmayBdownBwatchBgoodBcouldBwantBlastBhereByearsBuBthenBmakeBdidBwildfireBwayBhelpBbestBtooBevenBbecauseBhomeBdeathBcollapseBbombingBmassBhimBblackBamBthoseBneedBfatalBarmyBanotherBworkBtakeBshouldBreallyBpleaseBmh370ByoureBlookBlolBhotBpmBlegionnairesB4BrightB5BletBcityByearBwreckBschoolBnorthernBmuchBforestBbomberBwaterBsheBneverBreadBlatestBhomesBgreatBeveryB1BliveBgodBfearBanyBÛBunderBsaidBoldBfloodsB2015BgettingBatomicBwhileBtopBobamaBfeelBthatsBsinceBnearBflamesBeverBcomeBwhereBtheseBmilitaryBjapanBfoundBcontentBassBwithoutBweatherBmostBfloodingBfloodBdamageBwhichBshitBsBhopeBeveryoneBbeforeBstopBplanBmalaysiaBinjuredBhitB
evacuationBduringBdebrisBcrossBcomingBwildBwellBtimesBsinkingBoilBfuckingBcheckBcauseBweaponsBtruckBfoodBbloodyBalwaysBweaponBtheresBstateBlittleBinjuriesBfreeBwoundedBsummerBsmokeBsevereBredditBnextBmovieBiveBhesBfallBevacuateB	confirmedBbadBagainBthunderstormBsetBnightBnaturalBlooksBheatBfaceB
earthquakeBboyBwholeBuntilBthunderBthroughBsaysBpanicBoutbreakBmadeB	lightningB
fatalitiesBfamilyB	explosionBendBdestroyB
derailmentBairBwB	terroristBsurviveB	screamingBsaudiBrefugeesBrainBmurderBloudBlikedBhouseBgonnaBfailureBcollidedBbagBattackedB	ambulanceB70BwindBservicesBsaveBreportBmigrantsBheadBexplodeBchargedBchangeBbigBalsoBwreckedBwarningBupdateBrunBrescuersBreleasedBphotoBmassacreBinjuryB	hurricaneBhighBhailBfuckBdoesB	destroyedBbusBbloodB40BÛÒBwreckageBviolentBtwisterBtraumaBtragedyB	terrorismB	survivorsBsurvivedBsinkholeB	sandstormBroadBriotingBredBrealBputBpostBnationalBmissingB	landslideBkeepBgirlBdroughtBcurfewBbreakingBbagsBwhiteBtwitterBtonightB
structuralBspillBserviceBscreamedBrescuedBrescueBphoneBokBohBmosqueBlivesBhorribleBharmBgameBdustBdestructionBdelugeBdeathsBcrashedBcliffBcatastropheBboatBawayBaugustBareaB
apocalypseBwomanB	whirlwindBtraumatisedBstockBsawBruinBriotB
quarantineBkillsBislandBinvestigatorsBillBhostagesBhazardBdangerBcallB15BwomenB	windstormBthingsBsuspectBshowBreunionBquarantinedBlavaBheartBengulfedBdetonateBcrushB	collapsedBcameBbetterBbattleB
armageddonBairplaneBagainstBaffectedBuseBtrappedBthankBsunkBstoryBsendBpartBotherBmustBmudslideBmarketBiranBfamineBexplodedBelectrocutedBebayB	displacedBderailedBderailBburnedBbombedBblownBbabyBaroundBzoneBwaveBwannaBsureBsomeoneBscreamsBrazedBpowerBobliteratedBlongBlandBhundredsBheardBgroupB	flattenedBdrownBdoingBcareBbridgeBbaggingB9BwentBusedBtyphoonBtroubleBtornadoBthoughtBthingBriverB
respondersBpastBpandemoniumB	officialsBmeltdownBlotBleastB	inundatedBidBhostageB	hijackingB	hazardousBgoesBdrowningBdidntBdevastationBdemolishBcollideB
casualtiesBcalgaryBbangBanniversaryByetBwoundsBvolcanoBtsunamiBsueBstBsongB	somethingBshoulderBsecurityBprebreakBpossibleBpkkB	panickingBobliterationB
obliterateBmurdererBminuteBlightBletsBkillBisisBindiaBhijackerBhellfireB
governmentBfewB	evacuatedBdueB	detonatedB
desolationBcrushedBchemicalBblewBblazingBblastBannihilatedBairportB6BweekBupheavalBtryingBthreeBthanksBsoundBsoonBsirensB	rainstormBplaneBmusicBmakingBkidsBissuesBhalfBguysBfedexBdoneBdiedB
detonationBdaysBcycloneBcountyB	collisionBcausedBcatastrophicBbleedingB	beautifulB8BwordsBveryBtrafficBsouthBrememberBpolicyBplaceBnothingBnorthBmpBlongerBleftBisraeliBhellBfunBdrownedB
demolishedBcoolBbothB	bioterrorBbelieveB	avalancheBarsonBturkeyB	snowstormBsiteBshotBshootingBpicB
nowplayingBmediaBislamBinsideBhijackB
helicopterBfightBfatalityBfanBelectrocuteBdoesntBbuildingBbrownBbcBactuallyB16yrByesBwatchingBwaitBurBtellB	swallowedBseismicBsecondBrubbleBreÛBplansBmenBmemoriesBlineBlaBhorrorBhealthBhavingBfindB
eyewitnessBdelugedBchildrenBbushBanythingBalreadyBalmostBaircraftByourselfByeahBwhatsBtomorrowBsuchBstartBsideB	searchingBsavedBreactorBprobablyBplayBpersonBpeaceBoutsideBofficerBnearbyBnBmaybeBlostB	literallyBhoursBhearBfarBdieB
demolitionBdataBcrewsBconclusivelyBbusinessBamericanB20BÛÓBwestBwavesBteamBstreetBstayBsoudelorBreutersBmanslaughterBleatherBjobBhistoryBheyBfeelingBeyesB
everythingBdeclaresBdealBcasualtyBbodiesBamidBablazeB7B50B30B12ByouthBwontBwakeBtheyreBsupportB	stretcherBsameBriseBpickingBphotosBownBothersBorderBomgBokayBnameBmyselfBmoneyBmakesBleaveBlabBgtBgetsBflagBdesolateBcrisisBcenterBbookBblightBblazeBagoBabcB	11yearoldBwomensBtyphoondevastatedBtvBtrenchBtrainsBtexasBspaceBsirenBshesBselfBsaipanBreasonBrdBprettyBpickB	offensiveBmoveBmeekBmajorBmBlowBlordBhugeBhatBflashBfearedBfastBeffectBcourseBcountryBcontrolBclassBchildBchanceBcaughtBcalledBbioterrorismBbestnaijamadeBbecomeBbarBbannedBballBaugBannihilationBwrongBwinBusaBunitedBtownBtotallyBtoddlerBthoughBtempleBtakenBstandBspotBsignsBshipBpakistanBonlineBlevelBladiesBjobsBisntBhappyB	hailstormBfriendsBdiseaBdamnBcoupleBcaseBblueBbiggerBamericaBacrossB10ByoursBvillageBtryB	transportBtalkBseenBrussianBradioB	projectedBonceBofficialBneedsBnearlyBmountBmightBmayhemBinsteadB	hollywoodBhahaBguyBgunBgreenBfrontBfinallyBfavoriteBexpertsBentireBeastBdailyBcrazyB	computersBcoachesB	christianBchinaBblizzardBanyoneBaintBactionB25BvirginBvehicleBtruthBtrustBtakesBtBstarBsorryBrunningBrefugioBredditsBpoorBpainBmomBminersBmarksBlookingBknockBissuedB	insuranceBignitionBhousesBheavyBhateBhardBhappenedBglobalBgiantBgbboBflightBeyeB	emmerdaleBdriverB
devastatedBdBcostlierBcnnBcarsBcampBbeachBarsonistBangryBaloneBaddedB05ByorkBwonderBukBturnBtakingB
subredditsBsoundsBscaredBrussiaBrlyBreportsBreadyBquizBpublicBpropertyBpradeshBpplBplayingBpayBparoleBpamelaB	pakistaniBoutrageBniggasBnagasakiBmyanmarBmuslimsBmopBmadhyaBmadBlmaoBlearnBlargeBgovtBgiveBgemsBgaveBfuntennaB	fukushimaBformerBfilmBearthBdriveBdowntownBdogBcomesBclosedBcakeBbritishBbringBbbcBbBappearsB
aftershockB13B11ByoungBwowBworstBwavingB
washingtonBwantedBvsBviewBuponBtweetBtreeBtoteB	thousandsBthinkingBtheaterBsoulBskyBsignBshowsBshiftBseeingBseaBsceneBsafetyBrulesBrockBreportedBrBprayBplaylistBpatienceB
passengersBparkBnwsBnigerianBmorningBmomentBmodeBlistenBlikelyBlibyaBledBisraelBhumanBhiringBhandbagBhandBgopBgellerBgasBgalacticBfriendBfranceB	followingBfollowBfashionBenoughBelseBdrivingBdrakeB
donÛªtBdiseaseBdeclarationBcutBcoloradoBchinasBchileBcentreB
businessesBbiggestBbehindBbedBbayelsaBawesomeBarrestedBapolloBancientB70thB100BxBworkingB	wednesdayBwasntBunconfirmedBtwelveBturnedBtriedBthursdayBterrorBsuperBsexBsecretBsadBriskBrecountBproblemBpotusBplannedBparentsBoccurredBnumberB
neighboursBmphB	militantsBmiddleBlinkBlandingBlampBinfoBideaBholdingBgovernorBgladBgermsBfirefightersBescapeBebolaBearlyBdudeBdeepBdateBcreeBclaimsBcdtBbreakBaveBartBanthraxB60B16B12000BworseBworldsBwesternBwantsBwallBwalkBusingBtrueBtillBstartedBspringBsmallBsilverBseriousBprovokeBplayedBpartyBniceBmountainBmonthsBmodBmissBmishapsBmetalBmedBliesBlakeBladyBjapaneseB	involvingBinvestigatingBinternetBhourBhelpingBhaventBgtgtBfunnyBforceB	financialBfauxBfallingBexpectedBdownloadB	directionB
departmentBdenverBdeliversB
crematoriaBcoastBclimateBcaBbuyBbombsBalbumBalarmBaheadBaccountBabaB
ÛÏwhenByoullBwishBweddingBvictimsBudhampurBturkishBtravelBtookBtoldBtogetherBtodaysBthreatBsunBstoriesBsickBshotsBshareBseriesBsenseBseasonBsayingBsaltBroomBrecordBpurseBprepareBplungingBphilippinesBpassBpBoffroadBniggaBmembersBmeansBlouisBlondonBlivingBlinkedBkingBjoinBinternationalBinformationBhorseBhasntB	happeningBguideBguessBgroundBfutureBfridayBfourBfeatBfactBexpectBetcBdoorB	differentBdiesBdeBcompanyBcommentBcausesBcareerBcameroonBbeginB	australiaBarentBanswerBalongBallowsBalabamaBairlinesBactB600B2014B17BwtfBworryBwifeBweekendBwalkingBvictimBvButc20150805BtrumpBthoBtextBtestBteenBtaiwanBtabletBsyrianBstrikeBstraightBstatesBshortBshipsBshapeBsettingBsensorsensoBsearchBrockyBresearchBreadingBratherBrateBpressBpieceBphoenixBordersBonesBnycBminutesBmillionBmichaelBmeanBmapBluckyBkillingBjonathanBislamicB
internallyBinterestingBindianBheroBhappensBhappenBhairBgivingBgirlsBgeneralBfreakBfightingBfeltBfeelsBfatBfansBepisodeBenuguBeffectsBeachBeBdrunkBdoubleBdogsBdestroysBdesiresBdcBdanceBdanBcoverBcourtBcompleteBcoldBcloseBchangesBcarryingBcapturesBcableBbusesBbrooklynBboardBblockedBbitBbetweenBbanBashesBarianagrandeBapcBanimalBamazonBaliveBagreeBafricaB
absolutelyB2ndB24B2013B1stB19B18B0ByallBwroughtBworthBwindsBvoteBversionBupsetBunBtrentBtreesBtrainingBtracksBtotalB
technologyB	suspectedBstrongBstandardBstadiumB	speciallyBspecialB	sometimesBsoldiersBsnowBsmaugBsleepingBsingleBshB	selfimageBrouteBrossB	rooseveltBresponseBrepatriatedBremoveB	radiationBqueenBprogramBpreventB
populationBpictureBpetitionBpalestinianBofficeBncBnavyBmovingBmillionsBmetroBmeetBmediterraneanBmariansBleavingBleadBlateBkitBkillerBimageBholdBhoboBhelloBhandbagsBgunmanBgrowsBgottaBgoneBgoldBgamesBfullyBfrenchBfoxnewsBfeetBexpressB	examiningBeventBeveningBeitherBdrB	dangerousBdadB	currentlyBcranesBcopsBcolourBchicagoBcentralBcausingBcatBcannotBcanadaBcallsBbrotherBbrokeBboxBbeyondBbeatBbasedBavoidBanymoreBafghanistanBableB	abandonedB22B1980B14B06ByycBwwiiBwordBwonBwokeBvideosBusgsBusatodayBupdatesBtribalBtrfcBtotalingBthunderstormsBtensionBtearsBtargetBtalkingBsupposedBsundayBstuffBstudentsB	strugglesBstartsBstageBsquadBsposB	spaceshipBsonBsittingBsismoBshutBshouldntBshallBseeksBseekBscreenBsavingBsanBrunsBreliefBregionBreasonsBreBquestionB	presidentBpointBpileBperfectBpercentBpdpBpaulBopenB	municipalBmotorcyclistBmonthBmodifiedBmatchBmainBlotsBloseBlieBlessBleadingBlaterBkindaBkeepsBinsurerBindustryBimagineBiceBhuntBhospitalBhonestlyB	hilariousBftBforeverBforecastBfootballBfleetsBfiveBfearsBextremeB
everywhereB	emotionalBdrinkBdreamBdamagedBdaBcruzBcrimeBcreateBcookBconfirmsB
completelyB	communityBcomboBcoffeeBcancerBc130BbroughtBbrokenBboysBbooksBblockBbeganBbayBbabiesBattacksBaskBapplyBapBantiochBamongBallahB	accordingB911B4000B31B2011ByoB	worldnewsBwmataBwiredBwelcomeBwatchedBwarsBwaitingBusuallyBunitsB
understandBtropicalBtramBtontoBtipsBtime20150806BticketsBthxB
terroristsBsystemBstreetsBstreamBstoppedBsportsBspentBsirBsimpleBsignedBshowerBsharesBshameBsentBseemsBsecretsBsecondsBseattleBsaturdayB	satelliteBruleBroundBripBridgeBrichmondBreviewBreturnBrepairBreaÛBquiteBpressureBpracticeB	potentialBpoolBplusBpilotBpicsB
parenthoodBownerBoutlookBoriginalBopensBopeningBontoBoklahomaBoceanBo784BoBnursingBnahBmortalBmixBmineBmentionB
mediterranBmatterBmassiveBmaryBltBlovingBlovedBlistBlimitedBlilBleadsBleaderBkombatBkidBkickBjustinbieberBissueBirandealBinnerBincidentBimpactBhumanityBhobbitBhitsBhimselfBhighwayBhiBheresBhawaiiBhandsBgunsBgrillBgreatestBgovBgmBgayBforgetBfirefighterBfineBfinalBfalseBfacebookBfaanBexplorationBexitB
exchangingBepicBelBedmBearlierBdrinkingBdiscoBdirectionersBdeadlyBdavidBdarkBcryingBcostBcontinueB	concernedB	companiesBcommonB	civiliansBcivilianB	characterBcasesBcarefulBcanyonBcallingBburnBbuildBbrowserBbroBbringingBboughtBblogBbirthdayBbinB	bicyclistBbenBbattlefieldBarrivedBarBamazingB	afternoonBacresB	accidentsBabstormB90B500B26ByouÛªveByouveB	yesterdayBwxBwritingBworkedBwithinB	wildfiresBweirdBwaysBwarningsBwaBvuittonBvoiceBvisitBvietnamBunlessBuniverseBtripBtownshipBtongueBtoiletBthomasBtentBtankBsyriaBsurvivalBstupidBstoneBsteveBstartingBsportBsomaliaBsocialBsmhBsleepB	situationBshoesBsevenBsensorBseesBseatBscienceBsadlyBrubberBroofBroadsBresultBrespondBreplaceBremainsBrelatedBreduceBrecoveryBrealiseBrareBrailBquoteBquicklyBquickBpussyBprotectBproblemsBpriceBpreparednessBpowerfulBpostsBportlandBpopB	politicalBpointsBplayerBphotographyB	pathogensBofficersBnoneBneededBnatureBmumbaiBmsBmrBmomentsBmissedB	minecraftBmillBmileBmdBmayanB
manchesterB	malaysianBlossBlooseB	listeningBlionBlinkuryBlightsBliftedBladenBknewBkindleBjebB	itÛªsBitalianBiraqBinvoicesB
inundationBinjB	includingBillegalBi5BhwyBhwoBhttptcoqew4c5m1xdBhillBhearingBheadedBhaBguardianBgodsBglassBgenocideBgazaB	freakiestBfortBforcesBfogBfishB	feministsBfellBfantasyBfallsBfB
experienceBexchangeBexceptB	equipmentB	epicentreBepBendedBemBeconomyBeconomicBdroppedBdozensBdisneyB
discoveredBdiBcreatedB	continuesB
conferenceBcomputerBclutchBchurchBchoiceBchinaÛªsBchiefBcheeseBcharityB
canaanitesBburnsBbruhBbroadwayBbrainBboutBbombingsBbeyhiveBbetBbaseballBbankB	automaticBaudioB	attentionBaskingBaskedBarticlesBareasBarabiaBangelBalBahBaddBaccessB4x4B3gB1000BzombieByrsByaBxdBwriteBwouldntBwinsBwingsBwindowB	wheavenlyBweeksBwedBwasteBvinylBvineBveteransBvanBvalleyButcBupdatedBunlockedBunitBtypeBtwiceBtweetsBtubeBtroopsB	triggeredBtrailerBtourBtoughBtouchBtorontoBtomBtollBtiredB	threatensBthoughtsBtheatreBterribleBtenBtcotBswallowsBswB
surroundedBsurpriseBsunsetBsubjectBstyleBstruckBstormsBstoreBstopsBstepsBsteelBstealingBstarsB	standuserBsparkedB
soundcloudBsomebodyBsocietyBsnapBslightlyBskinBsizeBsitBsinjarB	sidelinesBsicilyBshelterBshadowBseveralB	septemberBseniorBsenatorBseBscaryBsatchelBsantaBsalemBsaleBsafeBrunwayBrowBrootBrocksBrnBrisingBriotsB	residentsBrepublicansBremindsBrememberingBreleasesBreleaseB	relativesBranBquranB	questionsBputinB
prosecutedBprophetmuhammadBprojectBprimeB
powerlinesBpollBpipeBpilotsBpcBpatientBpassingBparleysBparkerBpageB	operationB
officiallyBnytimesBnyBnoticesBnineB	nashvilleBmurderedBmtBmoviesBmoonBmoodBmonogramBmmaBmissionBministerBminingBminiBmindBmikeparractorBmetsBmessageBmentionsBmeetingBmedicalB
meatlovingBmaximumBmaxBmarketsBmaintenanceBmadinahBmacBlt3BlovelyBloopBlocalBlibraryBlevelsBlegacyBlearningBlawsBlaneBlackBknowsBkneeBkisiiB	kidnappedBjohnBjoeBjacksonBjackBiÛªmBitunesB	interviewB	intensityBindeedB
incredibleBiiBieBideasBhurtsBhuntersBhumansBhttptcoq2eblokeveBhttptcoencmhz6y34BhomelessBheroesBheldBheadsBhBgunfireBgtaBgottenBgivenB
girlfriendBgermanBfruitBfrBfoxtrotBforcedB	followersBfocusBfloridaBfloorBflatBfixBfiredBfemaleBfeedBfailedBfailBexpB
executivesBevilBeverydayBestimateBentertainmentBenjoyBenglandBenemyB
electronicBedtBeditionBeatBdyingBdubstepBdroneBdrillBdollarB	documentsBdnbBdistanceBdestinyBdefenseBdebateBcuzBcuteBcurvedB	criminalsBcrapBcramerBcouldntBcostsBcopBconsiderBcongressB
conditionsBconcertBcomplexBcoBclubBclickBchineseBchargingBceoBcarryBcardBcapsizesB
canÛªtBcaloriesBcalmBcalifBcBbyÛBbuttonBbutterBburstBbudgetBbroadBbrazilBbostonBblvdBblkB	blessingsBblessedBblamedBbellsBbearBawfulBavoidingBauthoritiesB
australianB
associatedBarriveBarmsB
apparentlyBappBanimalrescueB	amsterdamBakaBaimBafBadvisoryBadultBactionsBaccidentallyBaboveB5kmB53inchB4wdB35B300wB1945BzionistBz10ByoungheroesidByoudByobeByayBxboxBww1BwrittenBwouldveBworriedBworkersB	wonderingB	wonderfulBwoBwilliamsBwhosBwhetherBwetBweedBwearingBwealthBwarshipBwarnsB
viralspellBviolenceBvehiclesBusualBusersBupperBunveiledBultimateBughBtwiaBtrynaBtrulyB	treatmentBtreatB	trapmusicBtrapBtraditionalBtrackBtowelBtowardB	tornadoesBtitleBthusBthrowingBthrillerBthreateningBtheyllB	tennesseeBtemperedBtedBtechBteamsBteaBtbtBtaxBtagBsydneyBswimBsweetB	surprisedBstuartB
structuresB	structureBstreakBstrategyBstrangeBstrandedBstatusB	statementBstaffBspringsBspiritBspiderBspanishBsomehowBsolutionBsolarB
socialnewsBsmokingBsisterBsingingBsingBshopBshippingBshepherdBsharpBsexualBservedB	seriouslyBserialBsentinelBsarahBsacB
rÌ©unionBromanceBrollBrevealedBresultsBrestoreBrestBresponsibleBrespondsBrespectBresidentialB	reportingBrenoBremainBregisterBreducedBrearBrealizedBreadÛBrayBrapeBrallyBrainsBquestBqualityBputtingBpurpleBpumpB	protectorBprophetBprofileBpricesBpresBpovBportBpoliticsBplugBplotBplayoffsBplanetBplainsBplagueBplacesBpilingBpickedBphysicalBphillyBpersonsBpersonalBpeanutBpatrickBpartsBpaperBpantherattackBpakBownersBoptionBokwxBofÛBofferBnumbersBnukeBnormalBnewestBndBnavedBnationBnasahurricaneBnamedBmuseumB	murderousBmovesBmovedBmotorBmoBmlbBmilesBmigrantBmidoB
microlightB	mhtw4fnetBmessBmemoryBmemorialBmariaBmansehraBmanagerB
managementBmailBlungsBluckBlowndesBlowerBlovesBlorriesBlootingBlookedB
letÛªsBlegalBlawBlaughingBlargerBjusticeBjulyBjudgeBjewishBjetBjamaicaBjamBjacksonvilleBitselfBitalyBirBiphoneBinsaneB
injuryi495B	increasedBiiiBigersBicesÛBicemoonBi77BhungryBhttptcovvplfqv58pBhttptcoksawlyux02BhousingBhotelB	hopefullyBhipBheartsBhdB	guillermoBgtgtgtBgrowBgreyBgradeBgraceBgoogleBgolfBgoalBgermanyBgeorgeB
generationBgangBgabonBfreshBfreedomBfoxBfinnishBfightersBfieldBfestivalBfavBfakeBfacedB	extremelyBextraBexternalBexplosionproofBexperimentsBexactlyB	everyonesB	estimatedB
especiallyBeruptionBerrorBenergyBendsBemsBelephantB
electricalBeffortB	educationBeasyBdvdBdutyBdutchBdrugBdropBdronesBdrawnBdramaticB	djicemoonBdivingBdisruptsBdisneysBdietBdetailsBdepthBdemandBdegreesBdecidedBdebtBdealsBdarudeBdareBdamagesBcryBcrossedBcreamBcoversBcoveredB	countriesBcopilotBconsideringBconflictBconfirmBcombinedBcollisionnoB
collectionBclipBclearBcleanBclassicBcivilBcitiesBcinemaBciaB	christmasB
christiansBchargeB
charactersBcertainBcampaignBbuiltBbuffaloBbrideBbrakesBbowlBbossBbobBblindB
blackberryBbjpBbitchBbeÛBbelowB	beginningBbecomesBbecameBbeamBbb17BbatBbargainBb4B
australiasBasapBarticleB
approachesBanywayBantiBannualBanimalsB	anchorageBamongstB	americansBalpsBallowBaidBageB	afterlifeBafraidBafghanBadvanceBaddressBactsBactivityB	activatedBabuseBabcnewsB	97georgiaB731B3dB33B320B300B101BåÊBÛ÷politicsB	ÛÏtheBÛÏaBzoumaBzByrByazidisByBxpBwyBwriterBwoundBworstsummerjobBwoodBwitherBwitBwirelessBwireBwinstonBwindowsBwheresBwhaoBweveBwerentBwebsiteBwashBwarfightingBwantingBwalmartBwaimateB
vulnerableBvotejkt48idBvintageBvetsBvaluesBvalBvacantBvaButterlyButterBusesBusagovBuribeBunsuckdcmetroB	unlockingB
universityBunionBunderstandingBuglyBugandaBtwinBtutorialBturnsBturbineB
tubestrikeBtrucksBtrolleyBtripledigitBtribuneBtrialBtransitBtrackingBtoolsBtonsBtoddBtime20150805BthrowBthousandBtheyveBtheoryB
themselvesBthBtemporary300BtellingBtasteB	targetingBtampaBtableBswimmingBswanseaBsupremeB
summerfateBsufferBsucksBsuccessBsubsBsubBstudyBstressBstrategicpatienceBstirBstereoBstationBstarterBstandsBstabbingBstabBsprinterBspendBspeakingBspeakerBsparksBspainBspBsouthernB	southeastBsouthamptonBsourcesBsourceBsoupBsophieBsomeonesBsmithsonianBsmileBslowBslideB
slanglucciBskinnyBsixthBsixBsistersBsimulateBsilenceBshellBsetsBsenateBselfieBsectionBseanBscottBscoreB
scientistsBschoolsBschemeBscheduleBsatBsandiegoB	sanctionsBsaferBsaBrsBroseBrollingBrohingyaB	rockyfireBrockinBrocketBrobotsBrobertBrobBrisesBringBrightsB
ridiculousBriderBrideBrichardBrewardBrevealsBresponsibilityB
republicanB
reportedlyB	replacingB
rememberedBregularBrefugeeBredlightB	recommendB	recognizeBrecentlyBrecentBrecallBreapBrealizeBrealityBreachingBreachBrapidlyBrainingBrageBradarBracistBraceBpullsBpulledBpsB
providenceBprotestBpromptsBpromisesBpromiseBprogressBproducedBprobeBproBprisonBpriorBprintBprinceBprimaryB
previouslyBpreparedBpregnantB	predictedBpredictBprecipitationBprabhuBpovertyBpostedBpositiveBplsBplayersBplateBplantedBplantBplaguingBpitchBpipelineB	photoshopBphonesBperhapsBpepperBpbbanBpathB	passengerBparkingBpalmsBpalestiniansB	palestineBpaintingBpacksBpacificBpaceB	overnightBoutdoorBotrametlifeBoppressionsBoppositeBopinionBopenedBoopsBomfgBolympicBokanaganBoftenBoffersBoccasionBnwBnurseBnuBnpBnoticeBnobodyBnigeriaBnickiBnewyorkB	newlywedsBnetworkBnemaBnegativeBnbBnasaBmysteryBmuslimBmuscleBmultipleB
mtvhottestBmourningB	mountainsB
motorcycleBmotherBmorganBmonsoonBmindsBminBmichiganBmichael5sosBmiamiBmexicoB	messengerBmensBmemesBmeasurementBmeantBmealsB	materialsBmarylandBmarkedBmarkBmansionBmamaBmallBmajorityBmagicBmachineBlunchBlulgzimbestpictsBlowlyBlossesBlosingBlosBlongestBlonelyBlogoBlocationBloanBloadBlmfaoBlistedBlipB
lighteningBlezBleveledBlettingBlettersBlethalBlegitBleavesBleagueBleadersBlargestBkyleBknownBkindBkerricktrialBkeptBkcaBkarymskyBjstBjoyBjordanBjeansBjapÌnBiÛBiranianBinvalidBinternalBinternB
interestedBinterestBinsurersB	instagramBinnocentBincludeB
impossibleBimagesBigBidpBidkBidfireBhurtBhuhBhughesBhuffmanBhttptcozujwuiomb3Bhttptcowvj39a3bgmBhttptcoviwxy1xdykBhttptcothoyhrhkfjBhttptcolvlh3w3awoBhttpstcomoll5vd8ydBhostingBhostBhorsesBhorrificBhopBhonorsBholyBhollandBholidayBholeBhittingBhillsBhighestBhieroglyphicsBhereÛªsBheightsBheavenB
headphonesBheadingBhauntingBhatedBharryBharborBharamBgustyBgustsBgroupsBgriefÛªBgrenadeBgrazedBgratefulBgrabBgiftBgardenBgainedBgBfwyBfurtherBfundBfuelBfuckedB	frontlineBforwardB	forgottenBforgotBforbesBfollowsBflyBfloydsBfloodedBflewBflagsBfitsBfingersBfindsBfightsBfifthBfettilootchBfeedingBfbiBfatherBfateBfasterB	farrakhanB	fantasticBfamousBfalconBfaithBfactoryBexplainsBexpertBexistBexB
eventuallyBeventsBeuropeBestateBequalBepilepsyBenteredBenterBenjoyingBenglishBemotionsBelevatedBelemBelectricBelectionBeffortsB	edinburghBedenBebikeBeatingBeasilyBearringsBearnersBdtnBdryBdrugsBdroveBdrivenBdressBdragonBdoubtBdkBdissBdinnerBdiamondB	detectadoBdesireBdesignsBdesignBdescriptionBderailsBdeputiesB
dependencyB	democracyBdelaysBdegreeBdefendB	decisionsBdecisionBdecideBdearB	daughtersBdaughterBdamBdallasBcyclistBcurrentBcupcakeBcupBcrowdBcraterBcrashesBcrBcousinBcouplesB	counselorBcottonBcosBcordBcopycatB
constantlyBconcernsBconcernB	compliantBcommunitiesB
commercialBcommentsB	collapsesB	cofounderBcockBcloudsBcloudBclintonBclientB	clevelandBclevBclearlyBcleanupBclashBclaimBchillB	childhoodBchicagoareaBchelseaBchartsBcharlieBcharlesBchannelB	challengeB	certainlyBcellBcdcBcautionBcatsBcatchBcastBcarryiBcapturedBcaptureBcaptainBcapacityBcancelBcampusBcaliforniasBcalamityBbulletinBbugBbuckleBbrutallyBbrothersBbringsBbrakingBboxerBboundBbootB	bluetoothBbluejaysBblowsBblockingBblessBbitchesBbidBbffBbeyonceBbenefitsBbellBbeginsBbeesBbeerBbedroomBbeatsBbearsBbattlingBbatteryBbathroomBbaseBbandsBbakeBavertedBaverageBavBauthBaussieBauctionsB	attackingBatlantaB
assistanceBasianBasiaBarabianBappleBanybodyBanswersBankleBangelesBanalysisBamericaÛªsBamericasBallowedBallegedBalertBalcoholBalbertaBalaskaBagreedBadmitsBactiveB	activatesBaccusesBabilityBabbswinstonB852015B80sB800B6augB5pmB5000B48B3rdB39B370B360wisenewsB34B2pmB2pcsB29072015B23B21B2030B2009B18wB13000B125B1200B10thB02Bå¨BÛ÷extremelyBÛÏweBÛÏrichmondBÛÏhatchetBzaynByugvaniByorkerByepByemenByeaB	yahoonewsByahooBxxxBwwiBwrapupBworldnetdailyBwomBwndBwitterBwinterBwillingB
widespreadBwickedBwiBwhtBwhalesBweightBweeBweathernetworkBwearBweakB	waterwaysBwaterresistantBwashedBwarshipsBwarnedBwarneBwarnBwarcraftBwannabeBwallsBwalkerBvoterBvolgaBvladimirBvisitedBvirusBvirgilBvipBvillagesB	villagersBvidBvetBvestBvermontBverdictB	venezuelaB
vegetarianBvegasBvarietyBvampiroBvacationBusnwsgovBurgentBunsafeBunknownBuniformBunderwayB
undercoverBunawaresBunavoidableBudBuberBtyreBtyBtwinsBturningBtunnelBtuneBtsBtrubgmeBtrsBtripleBtriesBtrialsBtransportingBtransformationBtransferBtraffordBtrBtowardsBtorchBtoothBtoolBtonightsB	tomorrowsBtmpBtitanB	tinyjechtBtimelineBtilnowBtilBtiedBtieBticketBthruBthreatsBthirdBthinksBthinBthickBtheÛBtheydBtherapyBthatÛªsBtflbusalertsBtfBtestedB
terrifyingBtermsBtemperBtellsB	telegraphBteachingBteachersBtdpBtbBtaxiwaysBtargetsBtalksBtalentBtacoBtabÛBsystemsBsympathyBswingBswellsBswedenBsuvBsurvivorBsurgeBsurfaceBsurfBsurelyBsurahB
supervisorB	superheroB	sufferingBsufferedBstylishBstuckBstrikesBstressedBstomachBstewartBsteppedBstephenBstepBsteamBstaysBstartupBstarringBstandingBstampBspinningbotBspinningBspeedB	specimensBspecificBspecifBspearsBsoulsBsongsBsoldierB	socialismBsmokyBsmBslowlyBslowerBslipBslateBskirtBskillsBsizedBsittweBsittingÛBsitesBsinkB
silvergrayBsilentBsignificantBsightBshowcaseB	shouldersBshookBshockedBshockBshittyBshirtB	shipwreckBshelbyBsheddingBsharedBshanghaiBsexyB
settlementBsendsBsendingBseemedBseasonsBscreenshotsBscotlandBscoopitBscifiBschipholBscarsBsbBsavebeesBsassyBsandBsamplingB	salvationBsalmonBsafelyBsaddlebrookeBrwyB
rworldnewsBrunnerBruinedBroyalBromeBrollerBroleBrobotrainstormBrobinsonBrobertsBroanokeBrippedBridB	rickperryBrexyyB
revolutionBreviewsBrevenuesBreturnsBreturnedBretailBresultedB
restaurantB	respondedBrepublicBrepeatB	reopeningBreminderBrelaxBregretB
registeredB	regardingBrefusedBreflectB	referringB	recyclingBrecoverBrecipesBrecallsBrealtimeBrealmandyrainBrealdonaldtrumpBraynbowaffairBrangeBramagBrainfallBraidBradioactiveBquotesBqueensBpurchaseBpunBpullBptsdchatBptsdBpsychologicalBpsychiatricBprovideBprotestsBpropertycasualtyBpromptedBprofitBproceedsB
proceduresBprivateBpreviewBpresentBpremiumB	prematureBpreferBprayersBpostingBpossiblyBportionBpornBpopularBpondBpolandBpocketsBplaysBplanningBpizzaBpisgahBpillowBpiecesBphotographerBpetsBperiodBpeoplesB	penaltiesBpeacefulBpatternBpatchedBpatchBpassedBpanelBpamBpalinBpackBoutflowBosBornamentBoriginBorangeBoralBopusBoptionsBopsB
oppositionBoperBopenlyBoohBontarioBomarsBoliveBolderBolapBohioBoffensiveåÊcontentBoffensiveÛªBodeonBoddsBoddBoccursB	occupantsBocBoakBnytBnumberedBnuggetsBnsfwBnsBnriBnoticedBnoteBnoseBnoncompliantBnonBnoiseBnmBnikeBnieceB
nickcannonBnflBnewsintweetsBneitherBneedleB	necessaryBnbcnewsBnavblBnationsBnastyBnapBmusicianBmultiplayerBmullahBmsfBmovementBmountaineeringBmothBmoralBmodiministryBmodiBmodelsBmodelBmnpdnashvilleBmkxB
mittÛB
mitigationBmistakeBmirageBminsBminorityBminorBminajB	milkshakeBmilkBmilitantBmikeBmfsBmfBmetrofmtalkBmetricsBmethodBmercyBmercuryB	mentionedBmentallyBmentalBmemphisBmemberBmeltBmegaBmeatBmcilroyB	mcdonaldsBmattsonBmattersBmateBmatchesBmarvelBmarkerBmarineB	marijuanaBmanmadeBmajBmagnumBmaBm194BluisBlrtBloverBlosesBlonewolffurBlogBlockerBlockeBlockB	localÛBlocalarsonistB
listenliveB	listenbuyBlimitBlightingBlgbtBlessonsBlesbianBleoBlenBleisureBlegoBlegionB	legendaryBlegB
leadershipBlayBlaunchBlaughBlatelyBlasB	languagesBlangleyBlandfallBlaidBlBkuwaitBkurtschlichterBkurdishBkoreaBkomenB	knoxvilleBkingdomBkiernanBkickedBkeyBkenyaBkarachiBkalleBjustÛBjupitersBjulieB
journalistBjonBjokeBjoiningBjohnsonBjohannesburgBjoelBjimmyfallonBjesusBjazzBjaysBjayBjaxBjapansBjanBjamesBjailBjB	iÛªveBitunesmusicBirishBirelandBinvolvedB	integrityBinstantBinstallationBinstBinningBinkBinitialB
inevitableB	indonesiaBindividualsB
individualBindependentBimportedBillinoisBikBidpsBidiotBicymiB	ibookloveBianhellfireBianBhybridBhurryBhunkBhungerBhundredBhumidityBhumazaBhttptcoo91f3cyy0rBhttptconnmqlz91o9Bhttptcocybksxhf7dBhttptcocedcugeuwsBhttptcobbdpnj8xsxBhttptco9nwajli9crBhttpstcorqwuoy1fm4Bhttpstcodehmym5lpkBhoweverBhoustonBhorrorsBhoneyBhistoricBhillaryBhikeBhighlyBhigherBhelpsBheadsetBhatingBhatchetBhatcapBhashtagBharrybecarefulBharperB	happinessBhangB	hampshireBhamiltonBhamasBhaiyanBgunshotBgunsenseBguiltyBguidedBguardBgrowthBgrownBgroveBgroomBgrewBgregBgreecesBgrandBgrabbersBgpsB	gopdebateBgonBgoalsBgloriousBgivesBgerenciatodosBgeorgiaBgenuineB	generallyBgemBgateauB	gamergateBgameplayBfyiBfundsBfuelsBfuckinBfrozenB	frontpageBfreezingBfrankBfractionBfoulBformedBforgivenBfordBfootageBfootBflyingBflowerBfloatedBflamingBfknBfixedBfitnessBfitBfishingB	finishingBfinishBfingerBfilledBfergusonBfeministBfeelingsBfedBfeaturesBfdBfbB	favouriteBfashionableBfaroeislandsBfarmBfamBfallenBfairBfactsBfactorsBextendsBexploitBexperiencingBexperiencedB	expensiveB	expectingB	existenceB	excellentBexampleBevidenceBeuroBeuBespBerBeqBenvironmentalBenvironmentBenteringBenrtBenjoyedBengineBendingB	encounterBenBembroideredBemailBeliteBeightBegoBeggsBeggedBeeBeditorBedBebBeatenBeasternBearthquakesBearBdundeeBdudesBdublinBdualBdrivesBdrinksBdreamsBdragBdqBdorretsBdorretBdonateBdonaldBdomesticBdocBdjB
disgustingB	discoveryB	disastersBdisappointsBdisappearanceBdisBdirtBdiamondkesawnBdevalueBdetainedB
destroyingBdespiteBdesignedB
describingB	describesBdeptBdentalBdenmarkBdemBdeliverBdelayB
definitelyBdeclinedBdecemberBdecadesBdatingBdatBdarknessBdaeshBcuttingBcurbBcuntsBcumBcueBcubanBctaBcsxBcrypticBcrossesBcroatianBcriesBcricketBcrewBcrashingB	crackdownBcounterBcountB
correctionBcornersBcoolerBcontrolÛB	continuedBcontextBcontemplatingB	containedBconstructionBconsequenceBconquestB	connectorB	conditionBconceptBcommuteBcommunicationBcommitBcomedyBcombatBcolumbiaBcolorBcollision1141BcollegeBcobraBcoastalBcoachBcnbcBclothesBclosuresBclosingBclosesBclimbBclearedincidentBclaytonbryantBcladdingBcitizensBcircuitBcircleB	cigaretteBchunksBchosenBchokingBchoicesB	chocolateBchickenBchewingBchevyBchestB	chernobylBcheckedBchasesB	charlotteBchargesBchargerBchaosBchangedBchainB	centipedeBcecilthelionBcdBcbccaBcaveBcatchesBcasualBcasperBcarriedBcargoBcapsizedBcapitalBcandyBcameraBcamB	calum5sosBcakesBcaitlinBcabinBbypassBbuyingBbuttBburntBburiedBbunchBbullyBbuddysBbuBbtwBbtsBbsBbrushBbriefingBbrianBbreatheBbreaksBbreakingnewsBbraveBbrandBbradyBbpBbowlingBbountyBbottomBbotherBbornBboredBboreBborderBboomB	bookboostBbonesBbokoBboeingBbluedioBblowoutBblowmandyupBblowBblastsBblanketBblameBblakeBblahBbirthBbirdsBbioBbillingsBbillBbikerBbigamistBbewareBbetrayedBbesidesBbendBbelongsBbelongedB	beginnersBbeforeitsnewsBbeeBbecomingBbeardBbbcnewsBbb4spB
battleshipBbattersBbathBbasementBbarsBbarnBbarelyBbareBbarackobamaB
bangladeshBbandB
backgroundBaÛBawareBawBavenueBaussiesBatÛBattitudeBassholeBassemblyBarwxBartsBartistsunitedBarsenalBarrestBarmB
appreciateBapplicationsBapchB	apartmentBapartBaomsBanxiousBanxietyBannouncementBannaBangelsBamiriteBalthoughBalternativesBalrightBalexBalarmsBahhBaffectsBaddingBacuteBactualBactingBacidBaccusedBaccionempresaBabortionBabominationBabeBabandonB95B87B86B80B78B6thB69B56B55B45B43rdB41B375B361B300000B2usB27B2005B200B1620B150footB150401B150B103B0dayB09B010401B0104BåÈBåÇBåB
Û÷plotBÛ÷itÛªsB	ÛÏyouBÛÏstretcherBÛÏhannaphÛBÛ¢åÊdemolitionB
zippednewsBzionismBzimbabweB	zaynmalikBzabadaniByycstormBypresByouÛªreByouÛªllB
yourselvesByougovB
yesterdaysByellowByellingByazidiByankeesBxvBwwwcbplawyersBwweBww2BwudBwristBwowoBwouldnÛªtBworriesBwornB	worldwideBworksB	workplaceBworkoutBworkerBwoodsBwoodlawnBwolvesBwolfBwodBwocowaeBwmvBwmur9BwitnessBwippBwineBwindyBwilshereBwilliamBwideBwhoseBwhoopsBwhomeverB	wholesaleBwhoeverBwhoaB
whitehouseB
whitbourneBwhistleBwhippedBwheneverBwhatsappBwhaleBwftvBwfpB
weyreygidiBwestonBwenBwelfareBweighsBweekoldB
wednesdaysBwednB	wedaug5thBwealthyBweallheartonedirectionB	weakeningBwdBwceBwbioterrorismampuseBwbBwayneBwattpadB	watertownBwatersBwashingtonpostBwarriorBwarmingBwarmBwalterBwalkedBwalergaBwakingBwaitsBwaistBvotingBvotedBvoodooB	voluntaryB
volleyballBvolcanoåÊinåÊrussiaB	volcanoesBvodsBvitalBvistoBvisitsBvisionB
violationsBvinceB
vietnameseBvideoveranomtvBvidaBvictoryBvicinityBviceBvibezBviBveteranBversionsBversetheBveniceB	vegetableBveBvaultBvashonBvariousBvanuatuBvantageB	vancouverButc5kmButBushedBuserBus101BursBupwardsB	uploadingB	upgradingBupgradedBunrestBunrelentingBunrBunprecedentedBunnecessaryBunityBuniqueBunionsBuniformsBunhappinessBunfortunatelyBunfoldedBuncleBultimaluchaBukraineBtyposBtypesBtylerB
twovehicleBtwosB
twentynineBtweetlikeitsseptember11th2001BturkmenBtuningBtuneinBtuesdayBttesBttB
tsunamieshBtryoutsBtryoutBtrustyBtrunkBtrueloveBtroyB	troublingBtrollBtrinityBtrillionBtrickBtrendBtremorBtrekBtreepornB
treatmentsBtreatingBtreatedBtravisBtraverseB
travellingB	travelingBtransformedBtrancyBtragicBtradingBtouchedBtotesBtortureBtornadosBtorchingBtorB
topstoriesBtopsB
tomcatartsBtomatoesBtokyoBtlcBtixBtitanicBtinBtimkaineB	timelapseBtigersBtidesBtidalBthyroidBthyBthugginBthuBthrownBthrowingknifesB	throwbackB
throughoutBthronesBthroatBthrewBthreatenBthisizbwrightBthisiswhywecanthavenicethingsBthirstyBthirstBthighsBthiefBthetawniestB	therapiesBtheologicalBthemedBthemagickidrapsBthegameBtheeB
thankfullyBthankfulBthaBtgirlBtestsBtestingB	testimonyBteslasBtermBtensB
temptationBteethBteensB	teenagersBtechnologiesB	techniqueBtechnicaBteasesBtearB
teamstreamBteamhendrickBteacherBteBtchBtcBtaylorBtaughtBtattooBtastesB	tasmaniasBtargetedBtapeBtanzaniaBtanksBtanBtalkinBtaleBtakBtahoeBsÛBsyndromeBsymphonyBsxBswsBsweptBsweet2youngBsweaterBsweatBswearBswarmBsustainableBsuspenseBsuryarayBsurveyBsurucBsurroundingB	surrenderBsurgeryBsurfersBsupposeBsupportsBsupplyBsunnyBsunnisBsundaysBsummitB
summertimeBsuitBsuicidebombingBsuhBsuddenlyBsuddenBsubwayBstunsBstunnedBstudioBstudentBstuBstsB
strugglingBstronglyBstrongerBstrokeBstrikingBstrikesstrikesBstrikersBstrikerBstrictB	stressfulBstrengtheningBstoutB
stormchaseBstoresBstolenBstoleBstocktonBstocksBsticksBstickBstealBsteakBstatsB	stateÛBstartupsB	starbucksBstansBstandwithppB	standardsBstagetwoBstaceyBsrBsqueezeBsquareBsputnikB	spreadingBspotsB
sportwatchBspontaneouslyBsplitBspilledBspendingBspencersBspectacularBspeciesBspecialistsB
specialistBspeakBsovietB
southdownsB	soudelorsBsortBsorrowsBsoreBsooooBsoooBsooBsonsB	somewhereBsolveBsoloBsolitudeBsolidB	solicitorBsoldBsoftwareBsoftenzaBsoftBsocialmediaBsoccerBsoakingBsoakBsnuffBsnowdenBsnowballBsnipingBsneakBsndBsnapchatBsnacksBsmoothBsmilingBsmellsBsmellBsmashBslipsBslippedBslidesBslickerBslayerBslammedBskysBskylineBskippingBskillBsketchBskanndtyagiB
sixpenceeeBsixmeterBsisBsinglesBsinBsimultaneousBsimplyBsimonBsignupBsigningBsigalertBsidingBsidesBsiBshutdownBshowingBshowedBshoutingBshoutBshouldveBshoreBshooterBshootBshoeBshizuneBshiraBsheriffBshelliBsheetBsheeranBsheerBshedBshawBsharingB	sharethisBshantaeBshaniaBshamBshakingBshakesBshakerBsfBsewerBsettleBsetlistBserversBserverBserveBservantsBsequelBseptBsepBsellingBsellBselfiesB	selectionBseemB	seductionBsealBscrewBscreamqueensBscreamBscottwalkerB	scorpionsB	schwarberBscheduleÛBscaresBscareBscaleBsaturnBsatanBsarcasmBsandyB
sanctionedBsamesexBsamanthaturne19BsalvisBsalvadorBsakeBsailingBsailBsahibB
sacramentoBsa15tBs2gBryanB
rutherfordBrushBruinsBruiningBruebsBruddBrtrrtB	rtamericaBrssBrpicsBroyalsBroyalcarribeanBroverBroutecomplexBrottenBrotatorBrossumBrosesBroryBronnieBronaldoBrollsBrohnertparkdpsBroh3BrogueBrogerBrockbottomradfmBriyadhBriversiskiyouBrisksBrioBrihannaBrightwaystanB	rightwaysBriflesBridingBridersBrickBrichBreyB
revelationBrevBretweetBretroB	resultingB
restrictedBrestiveBrestaurantsB	responderB	respectedB	resourcesBreshapesBrequiredBrequestBrepayBrepairsBrepBreopensBreopenBrenisonBremovingBremovedBremovalBremorseBremixBreligionB
relentlessBrelationshipBrejectsBrejectedBrejectB	reinstateBreidsBreidBregionalB
regardlessBregardBrefuseB	redeemethB	recreatesB	recordingBrecommendedBreceiveBrecapBrealisedBreadsBreactedBreactBreachesB
rdhorndaleBrcmpBrbBraÌ¼lBrazingBraynorBraynerBrawBraungBratingBratesBratedBratBrappingBrappersBrapperB	rapidcityBrapBrandomBrammedBraisedBrainierBrainedBrailwaysBrailwayBrailsBraidersBragesBracingBquotedBquestioningBqueBquartzBquartersBqBpÛBputinsBpushingBpushB	purchasedBpuppyBpuppiesBpunjabB
punishmentBpunishedBpunctureBpulwamaBpullingBpublicizingBptboBptBpsalmsBps4BproudB
protestingB
protectionB
protectingBprotectdenaliwolvesBprosynBproposedBproperBproofBproneBpromoteBprojectsB	programmeBprofessionalBproductBprodBprivacyBpriorityB
prioritiesB	primarilyB	pricelessBpreviousB
preventionBpretendBpressedBpreservationBpresenceBprepperB	preparingBpreorderBpremonitionsBpremiereBpremierB
predictionBpredatorBpreciousBpreBpotentiallyBpostersB
postbattleBpossibilityBpossBportugalBpopularmmosBpopsBpoplarBpopeBpoorlyBpomoBpolitifiactB	policerunBpointingBpoemBpocketBpoBpmharperBpmarcaBplutoB	plummetedBpllolzBplentyBpledgeBpleasureBplatformBplantsBplanesBplacingBplacedBpjnetBpitsBpitchersBpissedBpinpointBpinerBpillowsBpillBpicturesBpickupBpickerelB
physiciansBphotosetBphiladelphiaBphilBphewBphaseBpharaohBphantomB
peterjukesBpertainsB
personallyB
perquisiteBpermitsBpermanentlyB	permanentB
performingBperformanceBperB
penningtonBpennB	pendletonBpeepsBpedestriansB
pedestrianBpeakB
peacefullyBpdx911BpcpsBpaysBpavedBpattonoswaltBpatriotBpathogenBpastaBpasswordBpartnershipsBparticularlyBparisB	paramedicBparadeBpantsBpanB	pakpattanBpakistannewsBpairBpaintBpaidBpaBovofestBovoBoverworkBoverlookingBoverloadBouvindoBoutfitBoutbidBoutageB	ourselvesB	otherwiseBothBosoBosbornBoriginalfunkoBorganizationsBorganicBorgBoregonBorderedBoppB
operationsBoperaBopedB
oooooohhhhBooohBooBonlinecommunitiesBoneselfBomegaBoksBokinawaBohhBoffsBofframpBoffrBofficesB	offers2goB	obviouslyBobsessedBobjectsBobispoBnylonBnwoBnvBnursesBnuffBnudeBntsbBnowhereBnovemberBnovelBnotificationsBnotedBnosurrenderBnostrilsB	nostalgiaB	northwestB	northlandB	northeastBnormanBnorBnopeBnonlifeBnoaaB
njturnpikeBnjengaBnjBnixonBnikeplusBnightsB	nightmareBnickcocofreeBnickBniallBnhsBnhBnewÛBnewzsacramentoB	newswatchB	newsaramaBnewbieBnewbergBneverendingB	netanyahuBnestleindiaBnepalBneighborhoodBneckBndaBnbcBnbaBnaziBnativeBnasasolarsystemBnarendramodiBnankanaBnanBnamesBnajibBnailBmÛBmyfitnesspalBmvBmuzzamilBmutualBmutantBmusikB	musiciansBmurdersB	murderersBmumBmultidimensiBmultiBmukilteoBmudBmsnbcBmoyoBmouthBmournsBmouldingBmotionBmoscowBmooresvilleB
montgomeryBmontetjwitter11BmonsterBmonkeysBmonkeyB
monitoringBmonitorBmomsBmohammedBmoderateBmockBmobileBmmmmmmBmlBmiyagiB
mistreatedBmississaugaBmissionhillsBmissilesBmissesBmishacollinsBmiseryB	ministersBminionsBminimehhBminhazmerchantB	milwaukeeBmigrantsÛBmidwestBmidsouthBmidgetBmidB
microsoftsB	microsoftB
microphoneBmicomBmichelebachmanBmicBmi17BmgmBmetlifeB
meteoearthBmetBmessiBmessedBmessagesBmercadosBmemeBmeinlcymbalsB
megynkellyBmeetsBmediumBmedievalBmedicineBmedalsB
mechanicalBmeaslesBmeaningBmcmahonBmchenryBmcBmatthewBmaterialBmatchingBmasterBmassmurdererB
martinmj22BmarshallBmarriedBmarqueiBmarlonB	marketingBmarinesBmariahBmarchBmapsBmanufacturedBmaniaB	mandatoryBmamataBmalikBmakethBmakerBmaintainBmagginoodleBmafiaBmadisonBlyricsBlyingBlukaBluchaundergroundBlubbockBlovedupBlousBlotzBloseitBlosdelsonidoBlorettaBloosesBloopingBlongtermBlongsBloneB
londonfireBlololBlollapaloozaB	logisticsBlogicBlockedBlocatedBloansBloadsBloadingBloadedBlmfaooooBllfBlizardsBlivingsafelyBliveonk2BlivedBlithBlitBlinksBlinesBlikesBlightningcausedBliftingBlifetimeBlifethreateningB	lifestyleBliberalBliableB	liabilityBlglorgBletterBletsfootballBlessonBlemonBlegsBlegislationBlegioBleedsBlearnedBleaksBleakedBleakBlayoutBlawsuitBlawrenceBlavenderpoetrycafeBlaurenBlaundryBlatimesB
latestnewsBlastingBlaoisB	lansdowneBlanguageBlanesB	landscapeBlandsBlanceBlamhaBlakesBlabsBlabelBkurdBks94BksBkraftBkowingB	koin6newsBknowingBknobBknifeBkneesBkmBkittensBkithBkissesBkissedBkissBkingsBkindnessBkindermorganBkillingsBkillersBkievBkiddingBkerryBkeratinBkendallBkeBkatunewsBkatrinaB
katherinesBkashmirBkarmaBkanyeBkadunaBjustmarriedB
justifyingB	justifiedBjurorsBjunkBjuneBjumpedBjuliedicaroBjuiceBjrBjpBjournalBjoseB	jonvoyageBjones94kyleBjonesBjonathanferrellBjointBjohnsBjohnnyBjklBjimBjhaustinBjewelryBjerryBjennerBjeffBjealousBjdabe80BjaptonBjamsBjamesmelvilleBjamaicaplainBjamaicaobserverBjailedBjacqueBjacketBjaBiÛªdB
ivanberroaBitsjustinstuartBitllBitemsBitemBisilBironyBironicBiredellBiraniansBipodBipadBinÛB	investingBinvestigationBinvestigateBinvestBinvasionB
interviewsB
interstateB
interlakenBintelligenceBintactBinsubcontinentB	inspiringBinspectionsBinsomniaBinsightBinsasB
innovationBinjuresBinfosecBinfinityBinfernoB
infectiousB
inevitablyBinecBindoorsBindieB
indiannewsBindiB
incrediblyBincreaseBincaseBincBimpulseBimproveB
impressiveB	impressedB	importantB	imperfectB	impendingBimpactedBimmediatelyBimdbBimaginedBimaBillustrationBihhenBignoredBignoreBidolBidisBidentitytheftBidcBiconicBicelandreviewBi95Bi580Bi405Bi10BhÛBhysteriaB	hypocrisyB	hyderabadB
hurricanesBhurricanedolceBhumbleBhumanconsumptionBhugB	httptÛBhttptcozevakjapczBhttptcozdtoyd8ebjBhttptcoyduixefipeBhttptcoydetwgribkBhttptcoxssgedsbh4Bhttptcoxpfmr368ufBhttptcovz1irh0nmmBhttptcovxvfaeey0qBhttptcovam5podgywBhttptcov3azwoamzkBhttptcouoozxaus26BhttptcoskqpwsnoinBhttptcosdgoutwntbBhttptcosaf9mosksnBhttptcoroi2nsmejjBhttptcopyehwodwunBhttptcopvmr38lnvaBhttptcopo19h8ycndBhttptcophixznv1ynBhttptconmfsgkf1zaBhttptcomg5eajelulBhttptcomfckpvzfv8Bhttptcom5kxlpkfa1Bhttptcom203ul6o7pBhttptcolxtjc87klsBhttptcolwwojxttivBhttptcojhpdssvhveBhttptcoj5mkcbkcovBhttptcoio7kuug1uqBhttptcoiikssjgbdnBhttptcoidmhswewqwBhttptcoi27oa0hispBhttptcoeysvvza7qmBhttptcoedyfo6e2puBhttptcodydfvz7amjBhttptcoct9ejxolpuBhttptcoc1h7jecfrvBhttptcobtdjgwekqxBhttptcoafmkcfn1tlBhttptco7hanpcr5rkBhttptco3tj8zjin21Bhttptco3sicroaanzBhttptco0wratka2jlBhttpstcowudlkq7ncxBhttpstcolfkmtzaekkBhttpstcoe8dl1lncvuBhrBhousedBhottestBhostageamp2B	hospitalsBhopingBhoodBhonestBhomsBhomieB
homeownersBholmgrenBholedBholdsBhoesBhoeBhockeyBhmuBhmmBhiroshima70BhiphopBhintonBhimalayaBhidingBhideBhiddenBhewBherosBheroinBhermancranstonBhenryBhelplineBhellaBhelicoptersBheavenlyBheartwarmingB	heartlessBhearthstoneBhealthyB
healthcareBhealingBhealB	headlinesBhazmatBhaveÛBhatredBhatchetwieldingBharwichBharmkidBhardyBhardlineBhardcoreBhappilyBhandlingBhandleBhamburgBhaltBhallBhahahahaBhahahahBhahahaBhahahBhahBhadntBhackersBhackBhabitsBgymBgustBgunmenBgumBguestB
guaranteedBgtiiBgrowingupblackBgrowingBgroundsBgrenadesBgreenwayBgreenharvardBgreekBgreatbritishbakeoffB	graveyardBgravelBgrassBgrandpaBgrandeurBgrandeBgpmBgovernmentsBgoulburnBgorgeousBgoodbyeBgolemB	goldsteinBgokuBgofundmeBgoatBgnBgmtBgmmbcB
gloucesterBglobeBglobalwarmingBglinkBglimpsesBglennBglassesBgiveawayB	gilbert23BgigBgifBghostwriterBggB	geometricBgeniusBgenevaBgeneralnewsBgelBgearBgdBgaysBgawxBgatesBgarfieldBgardensBgarbageBganderBgainingBgainBgadgetsBgaBfuryBfunctionBfukushimatepcoBfuelingBfruitsB
frontlinesBfrigginB
friendshipBfresnoBfreshmanBfreakingBfranklinBfranBfragileB
foxnewÛB
fouseytubeBfourthBfosterBforÛBfortuneBfortunatelyBforthBforsureBforgiveBforexBforeignBforbidBfoolB	foodscareBfollowerBfolkBfoldBfmBfluidBflowBfloridaÛBfloatBflipBfleeingBfledB
flashbacksBfixingBfistBfireÛBfireyB	firetruckBfiremenBfirefightingBfirBfinishedB	financingBfillBfilesBfileBfiftyBfifa16BfierceBfiegBfiB
fevwarriorBfestBferriesBfergusonÛªsB	fergusonsB
fennovoimaBfellaB	feinsteinBfederalBfeaturedBfeastBfavsB	favoritesBfavorBfaultyBfathersB
fatburningBfatallyB
fatalityusBfartBfamiliaBfallacyBfairyBfairfaxBfadingBfacingBfacilityBfabricBextinguishedBexterminateB	extensiveB	extensionBextenderBexploresBexplodesBexplainBexpandBexitedB
exhibitionB	exhaustedB	exercisedB	executingBexcuseB	exclusiveBexcitingBexcitedBexcBexaminerBevolveB	everytimeB	everybodyB
evansvilleBevanstonB
evacuatingBevB
eurotunnelBeudrylantiquaBetisalatBetBesteemedBeshBescapingBescapedBergoBequateB	epicenterBenÛBenvw98BentrepreneurBenrouteBenormousBenhancedBengvausBengageBenemiesBenduresBendorsesB
endangeredBencouragementBencoreBenabledBemptyBempireBemperorBemotionallyBemergesBemergBelsaBelliottBelkhornB	elizabethBelementBelectricityBehBegyptBefakBef5BedgeB	economiesBeastwardBearsBearningsBearnedBearbudsB	dystopianBdyeBdwarvesBdvcBdurantBdunbarBdumbBdukesBducksBduckBdtBdsBdryerBdrumBdroppingBdroidBdriversBdriftingBdrewBdressedBdreamingBdrawBdramaBdozenB
downstairsB	downpoursBdownfallB	douchebagB
doublecupsBdothrakiBdormanBdopeyBdopeBdoorsBdonå«tBdolphinBdollarsBdocumentBdoctorsBdoctorBdockBdmplBdmBdlhBdividedBdistrictBdistinctB
disruptiveBdisplaceB
discussionBdisappearedB	disappearB	directorsBdirectorB	diplomacyB
dijkÛªBdignityBdigitsBdigitalB	difficultBdiesisB	dickheadsBdickBdiarrheaB	diaporamaBdiamorfiendBdiabloBdeyBdevilBdeviceBdevelopBdetroitBdestructiveB	destroyerBdespairBdeskBdeservesBdeserveBdescriptionsBderbyBdeputyBdenyBdenierBdenialBdenaliBdemsBdemonB	democratsBdemiBdeleteBdelayedBdelBdefineBdefectsBdefBdeedsBdecorBdeclaredB
deckÛBdeckBdecentBdebatequestionswewanttohearBdealbreakerB	deadliestBdaytodayBdavidcameronBdatesBdarkestBdanteB	dannyonpcBdanisnotonfireBdanielsBdancesBdambisaBdamagingBcyclistsBcyclingBcustomsB	customersBcustomBcusterBcusB	curiosityBcuffBctdBcsBcrustyBcrudeBcrownsBcroatBcriticalB	cripplingBcriminalBcrimesBcriedBcricketsBcreditBcreatingBcreatesBcraneBcracksBcrackingBcrackBcoyotesBcoworkerBcowboysBcowBcoverageBcousinsBcourtsBcoursingB
countynewsBcountrysB	countlessBcouldveBcostlyBcorrespondentBcorrectBcorpBcorleonedabossBcoreyBcoralBcopycatsBcopingBcopeBcookingBconversationsBconversationBcontructionBcontrollersB
controlledBcontributingBcontrastBcontractBcontinuallyBcontainBcontactBconstantBconsoleBconservativeBconsentBconnectorconnectoB
condolenceBcondemnationBcondemnBconcreteB	concludedBconBcompoundBcomplicationsB	completedB
complaintsBcompeteB
comparisonBcompBcomoB	committeeB	committedBcommerceBcommandBcolludedB
collectiveBcoleBcolderBcolBcoincideBcodBcockpitBcocaineBcoatBcoasterB	cnewsliveBcmonBcmBcluelessBclosestBcliffsBcleverBclericBclearedBcldBclaimedBcjoynerBcivilizationBcityofcalgaryBcitesBcircusB	chronicleBchristieBchristBchrisBchooseBchillsBchiefsBchickBcheyenneB	chevroletB
chesttorsoBcherylBcherryBcherokeeBchaseBchannelsBchangingBchancesBchanBchampionshipB
challengesBchairsBchairmanBchairBcfcB	certifiedBcertificateBcentersBcensusBcementBcelebrationsBcbsbigbrotherBcbsBcbcBcawxBcatchingBcastleB	casperrmgBcashBcartBcarrBcarpetBcarlosB	caribbeanBcareersBcapeBcanvasBcannonBcandleBcancersBcancelsB	cancelledBcanadasBcampsBcampingBcameoBcalumetBcairoBcafireBcafeBcadfyiBcablesBcabBc4newsBbÛBbwpBbutterfingerBbusyBbundleBbumperBbumpBbullseyeBbulletsBbuildsBbufferBbubbleBbrunetteBbruiseBbrookeBbronxBbritonsBbritneyBbriskBbrightonBbrigadeBbriefBbrickBbrewingB	breathingBbreadB	brazilianBbradleybrad47BboxingBboundsBboundaryB
boundariesBbottleBboiseBbodysBbodybaggingBbobcatsBbmwBblutz10BbluntBbluesB	blueprintB	bloombergBblocksBblizzheroesBblizzarddracoBblamesB	blackpoolBblacklivesmatterBbitsBbitesBbiteBbitcoinBbistroB
birminghamB
biologicalBbillsBbillneelynbcB	billboardBbicyclesBbicycleBbicepBbetzBbetsB	bethlehemB
bestsellerBbengalBbeltBbellyB	believingBbeliefsBbeliefBbehalfBbeggingBbeclearoncancerB
beckarnleyBbeautyBbeanBbattlesBbattingB	batteriesBbatteredBbasisB	basicallyBbashesBbaruchBbarrierBbanquetBbanginBbanerjeeBbancodeseriesB	baltimoreBballsBbalanceBbakeofffriendsBbailBbagoB	backyardsBbackyardBbackupBbacksBayBawwwwB	awarenessBawaitsBavoidedBaviationBavengersB	availableBautumnBautoinsuranceBautisticBauthorB	authenticBaustraliaÛªsBaustinBaustBauntBaudienceBauctionBaucklandB
attractionBatticB
attendanceB
attemptingBattemptBatmosphericB
atmosphereBatmBatlanticBatkBathleteB	astrologyBassociationB	assistingB	assistantBassholesBasleepBasksB
askcharleyBasicsBashleyBashayoBashBasfBartistB
artificialBarsonistmusicBarsBarrivingBarnhemBarmoryBarmedBarizonaB
ariaahraryBargumentBareaÛBarchitectureB	architectBaquariumBaptBaprilBapprovesBapprovalBappropriationBapproachingBappreciatedBappointmentBappliesBapplaudBapollobrownBapocalypticB
apartmentsB	aogashimaBanywaysBany1BantonioBannoyingB	announcesB	announcedBaniBangerBandrewBandorBamesB
ambulancesBalotBaloisBallyBalloyBalliesB
allegianceBallegationsBaliensBalienBalexbelloliBalertsBalbanyBalarmedBakBagreesB
aggressiveBagesBagentsBagencyBagBafricansBafpBaffiliationB	affiliateB	affectingBafcBadvisedBadviceB
advertisedB
adventuresBadvancedBadmitBadministrationBadminBadditionBadamBadBactorB
activitiesBactedBacquireBacousticBachimotaB
accustomedBaccountsBacceptBaccBacBabusingBabuseddesolateamplostBabusedB	abubaraa1BabsoluteBabombBabiaBabc7BabbottBaaBa5Ba1B9pmB9newsgoldcoastB9amB96B90thB90blksamp8whtsB9000B900B8thB8pinB84B83B77B75B72wB64B61stB60mphB60000B5thB5sosfamB5sosB5sB5cB57B548B530B53B4thB4playthursdaysB4kmB4500feetB43B429cjB3mB	3inspiredB3942B360B36B32B2k15B299B29B28B23kmB235409B233liveonlineB21aB211023B
2082676773B2016B20150805B2012B2010B2008B2000B1999B1998B1986B1979B1976B1974B1970B1965B1943B1916B	18wheelerB1880B180B175225B1716B166B15thB15kmB143B140B12newsB118B1100B10kmB109B1061thetwisterB1038pmB1030B100000B075B0700B070B06jstB0306B015025B005225BåÊiB	åÊfedexBå¬onlyB
å©danielBå¤Bå£9Bå£6bnBå£27900endBå£150B	å£100bnBÌüBÌÑ1BÌÑBÌ¢BâÂBÛ÷weBÛ÷vulnerableÛªB	Û÷theBÛ÷secondÛªBÛ÷ransomwareÛªBÛ÷nuclearBÛ÷notherÛBÛ÷muslimBÛ÷minimumBÛ÷letÛªsBÛ÷leavesBÛ÷itBÛ÷instituteB	Û÷illB
Û÷hoaxBÛ÷hijackerB
Û÷heatBÛ÷hazardÛªB
Û÷goodB
Û÷foodBÛ÷firstÛªBÛ÷facelessÛªBÛ÷facelessBÛ÷exceptionalÛªBÛ÷emBÛ÷devastatedÛªBÛ÷britishB
Û÷bombB
Û÷bodyBÛ÷badgesBÛ÷avalancheÛªBÛ÷aminoBÛ÷allooshB
ÛÓkodyB	ÛÓherBÛÒåÊcnbcB	ÛÒtheBÛÏymcglaunBÛÏthehighfessionsBÛÏthatÛªsBÛÏsippinÛªBÛÏplansBÛÏpartiesBÛÏnumbersBÛÏnobodyB
ÛÏmakeBÛÏmacdaddyleoBÛÏlordbrathwaiteBÛÏlolgopBÛÏleoblakecarterBÛÏleejasperBÛÏkeitsBÛÏiB	ÛÏforB
ÛÏfdnyBÛÏdylanmcclure55BÛÏdetonateÛB	ÛÏcatBÛÏbbcwomanshourBÛÏbbcenglandBÛÏbasedgeorgieB	ÛÏallBÛÏairplaneÛBÛÏBÛ¢Û¢ifBÛ¢imBÛ¢BzzzzB	zxathetisBzurichBzumiezBzssBzrnfB	zourryartBzotar50BzoomBzonewolf123B
zonesthankBzonesBzombiesBzombiefunrun2014B	zomatoausB	zojadelinBzodiacBzmneBziuwB	zippolineBzippersBzipperBzippedBziphimupBzipBzionistsB	zimmermanBzimmerBzicacBzhenghxnBzhejiangBzerosBzeroBzergeleBzeno001BzenandemcfenBzehrsBzealBzaynmaiikistBzarryB	zarharzarBzarB
zamtriossuBzamanBzakuunB	zakbagansBzaibatsunewsBzachzaidmanBzachBzacbBzaatariBz3kesk1ByzfByyjByyesoB
yycweatherB	yycfringeByuviByuukoByuppiesByupByunita99ByumikoByumByukisByugByuanBypgByoÛB	youÛByouÛByouuuByoussefyamaniByourboyshawnByounooneB	youngÛB	youngsafeByounginsByoungerampgrosslyByoungerByosemiteB	yorkshireByorByonewsByolkB	yolandaphByogurtByogaByoenisBymcglaunB
ykelquibanB	yiraneuniByikesByieldByiayplanByhngsjlgByhB	yeyeulalaByessumByeshayadByennoraByemenisByellowsByelllowheatherByelledByellByehudaByeehawByeatByeaahhBydayBybtheprophetByazidishingalgenocideByardB	yamashiroB	yamaguchiByahootvByahoonewsdigestByahoofinancehopeB	yahoocareByahoo7ByahistoricalB	yagitudehB	yaboilukeB	xylodemonBxxhjescBxviiBxtra1360BxraysBxpostBxoxoBxojademarie124BxoBxmenBxmasBxlBxkdrxBxiiBxhnewsBxgninfinityBxfilesBxeniBxelaBxekstrinBxdojjjjBxdescryBxboxoneBxb1BxaviermarquisBxavierBx37bsBx2Bx1441Bx1434Bx1411Bx1402Bx1392Bx1386BwÛBwzbtBwyrmwoodBwyouBwyattb23BwxkyBwxiiBwxiatvBwwwbigbaldheadBwwwBwwpBwwexdreamerBwwaBww3BwwBwutB	wuglinessBwtwitterBwtonyBwthBwtcB	wsvr1686bBwsocBwsoaringBwslsBwsjthinktankBwsjBwsazbrittanyBwrougBwroteBwrongwayBwrongpersonBwrongdejavuBwrkedBwritingtipsBwritersBwritebothfistsB	wristbandBwrightsboroBwrestlerB	wrestleonBwreakBwrappedBwrapBwraithBwrBwqowBwpt994BwpsBwpoBwowtheBwowsavannahBwoutBwoundedÛBwoundedpigeonBwouldelectrocuteB	worthlessBworstoverdoseB	worsteverBworshipBworsenBworseitsBworryingBwormBworldwatchesfergusonBworldvisionBworldpayBworldoilBworldnetdailyhomosexualityB	worldletsB	workspaceBworkdBwordpressdotcomBwordkBwordingBwoooooooBwoodwardBwoodlandBwoodenB
woodchucksBwooBwondersBwonderousallureB	wonderkidBwonderfullyBwomppppBwomenÛªB
womengirlsBwomemBwombBwomanÛªsBwolterBwolforthBwokenBwoesBwoahBwnwBwnukesB
wniagospelBwnBwmiddleBwmBwldB
wlandslideBwkrnBwkndBwkBwizardBwiwnpfxaB
witnessingB	witnessesB	witnessedBwithåÊannihilationB	withstandB	witheringB
withdrawurB	withdrawsBwitchBwishlistBwishingBwishedBwiselyBwiseB	wisdomwedBwisdomBwisdcBwipesBwinnipegBwinningBwinnerBwinikBwingersBwingedBwingBwinechatBwindwakerstyleBwindstormåÊinsurerBwindstormfollowBwindsorB	windows10BwindowgatribbleBwindmyBwinditsBwin10BwimpB	wimbledonBwilsonsBwillowBwillisBwillingheartedB	willieamiBwillianBwillhillbetBwildwestsixgunBwildlookingB
wildlionx3BwildlifeB
wildhorsesBwildenB	wikipediaBwiiBwifiBwifekidsBwiedemerBwidthBwidoutBwiderBwidda16BwickettBwhyorBwhvholstBwhoÛBwhopperjr760BwhodBwhocaresBwhittBwhitewashesBwhistledBwhistleblowerBwhiskeyBwhippenzBwhipeBwhimsyB	whileÛBwhereasBwherB
whensoeverBwhelenBwheelsioBwheelBwhedonesqueBwheatleyBwhatevsBwhateverBwhatcanthedoBwhashtagB
whackamoleBwhaBwfriesBwfaaweatherB
weÛªveB
weÛªreBwexlerBwewsBwewBwestÛªsBwestwardBwestsBwestministerB	westmarchBwesterosnahBwesterosBwesterncanadadroughtBwestchesterBwesleyloweryBwesleyBwereonadesolateplanetBwerenÛªtBwengerBwendellBwelshninja87BweloveyoulouisBweloverobdyrdekBwelovelaBwellsB	wellknownBwellgroundedBwelles7BwelladjustedBwelcomesBweirdoBweiqinBweightsB
weightlessBweighBweepBweeklongBweekendsBweeiBweeblyBwednesdayÛBwednesBwednedayBwedgieBweddinghourBwebsitesBwebinarBwebBweatherstayB	weatheritBwearsBweaponxmusicBwealilknowaB
weaknessesBweaknessBwdyouthBwdymBwdtvB	wderailedBwcwBwctv35B	wccorosenBwcBwbuBwbreB	wbcshirl2Bwbc2015B
waziristanBwayÛyeahBwaywardBwaynesteratlBwayiBwayfieldstoneB
waveÛªBwavedB
wattys2015BwattleB	waterÛBwatersafetyB
waterproofBwaterfurBwaterboardingBwatchthevideoBwatchoutBwatchinBwatchesBwastingBwastenoxiousB
wastelandsBwastedBwasnÛªtBwasnamp8217tBwasillaBwashingBwashardBwaseembadamiBwarÛBwarzoneBwartimeBwarthenBwarsgoddessBwarriorcordB	warrantedBwarraBwarpedBwarningwildBwarnings900037BwarnerrobinsBwarnedÛBwarmthB
warmbodiesBwarlordqueenBwarfareBwardensBwardBwantmyabsbackBwanotherBwankBwanderB
waltdisneyB
wallÛBwallybaiterBwalesBwakhoBwakeupfloridaBwaiverBwaitedBwaimeaBwahpetonB	wahhabismB
wageÛªB	waferthinBwackosBwackoesBvÌdeoBvÛB
vzwsupportBvvormB
vuzuhustleBvulnerabilityBvulneraBvulnBvtcBvromanBvra50ÛBvotesBvotersB	vosloorusBvortexBvoortrekkerB	voodoobenB
volunteersB	volunteerBvoltaireB	volfan326BvolcanotornadoBvolcanodiscoverBvolcanicBvoidBvoicesBvodkaBvocalsBvocalistBvocalBvmasBvj44BviÛB	vixstuartB
vixmeldrewBvividBvivianunhcrBviviangiangBvivianBvivaargentinaBvitesseBvitalyB
vitalvegasBvitaBvistingBvisitingB
visionzeroBvisibleB
visibilityBvisageBvirtualBvirBviperBviolinBvioletsBviolentfeminaziB	violatorsB	violationBviolatedB	vinustripBvinnieBvinesBvincentBvimvithBvimeoBvillicanaaliciaBvillaB	vilelunarBvikingsBvigilsBvigilentBviewsBviennabutcherB	videogameB	videoclipB
victoriousB	victoriasB	victorianBvictoriagittinsB
vickybrushBvichardyBvibratesBvibrateBviabBvhullBvhsB
vgbootcampB	vets78734BvestmentBvesselsBversusBveronicadlcruzBvernB	vermilionBverhoekBvergilBvergeBverdeBventureB
ventilatedBventBvenomsBvenetoBveldfestBveldBveinsBveilBveggiesB
vegetablesBvegassolitudeBveganBvegBvectorBvaxshillBvastB
vassalboroB
varagesaleBvanpoliBvannuyscouncilB	vanishingBvanishedBvanillaBvanessasBvanessaB
vandalizedBvancouveråÊislandBvampiresBvalueB
valuationsBvalleywxB	vallerandB
valentinesBvaleB
valdes1978BvailBvaiBvaginaorcakeBvagersedollaBvaccinesBvaccineB	vacanciesBvacaBvabengalBv452BuÛBuxBuvopwzBuveBuvButvButpButopianButilizedButilityButicaButfireButdButc3kmButc20150806B	utahgrizzB
utahcanaryBuswarcrimesBuswB	uspacificBusmntBushankaBusgBusfsBuselessBusdotBusbushBusatodaynflBusatBusarmyBusamisanBusamaBusagiBusageBurufusanraguBuruanBurselfBurogynBurineB	uriminzokBurgentthereBurgBurbanisationBurbanfashionÛBurbanBuraniumBupwindstormBupwardBuptownjorgeBuptownBuptotheminuteBupstairsB	upsettingB	uprootingBuprootinB	upliftingBupiBuphillBupgradesBupdatemeBupcomingBupahBupaBunwomenBunwarrantedBunwantedBunuBuntoBuntillBuntameddirewolfBunsurprisinglyBunsureBunsuccessfulBunstoppableBunstableBunsignedB
unsensiblyB	unsecuredBunrecognizedBunrealtouchBunrealB
unpreparedBunpredictableBunplugBunpackedB
unnewsteamBunloadsB
unlicensedBunknowinglyBunivsfoundationBuniversityoflawB	uniteblueBuniteBuninvestigatedBunimpressedBunimaginableBunhingedBunhealedBunharmedBunhappyBungodlyBunfortunemelodyBunfollowBunfoldBunfmlBunfairBunexplainableBunendingBundoneB
undetectedBundeservingBunderwriterseniorBunderwriterB
understoodBunderstandÛBunderstandableBunderpassesB
underminedBundergroundrailraodBundergroundbestsellersBundergroundBunderestimateBuncoverBuncontrolledBuncontrollableBunconsciouslyBunconsciousBunconditionalBuncommonBuncomfortableBunclesBuncertaintyeconomicB	uncertainBunbelievablyBunawareBunauthorizedBunarmedBunaddressedBunableBumntuBummBumbrellaBumBuluruB	ultimatumBullmanB	ukÛªsBukrainesBuknewsBukfranceBukfloodsBuhmmmmBuhhhhhB
uglypeopleBuglyamesocialactionBugliestBugcBufo4ublogeuropeBufnBudomBudhampuragainBuchicagoBuabstephenlongBu2BtÌüpBtÛBtyroneBtyrantB
typographyBtypingBtypicalBtyphoonÛB
typewriterBtyleroakleyBtyarBtxtBtxlegeBtxBtwxB	twoptwipsBtwooutBtwitsandiegoBtwitchBtwistBtwillB	twilightsBtwiB	twentysixB
tweetstormBtweetinglewBtweetingBtweetedBtweet4taijiBtweenBtwcnewsBtwainB
tvshowtimeBtvjnewsB	tutorialsBtuskyBturnerBturnedonfetabooBturdnadoBturbojetBtunisianBtunisiaBtunisBtuneswggBtunedBtunasBtumblrBtumblingBtumblesB
tulowitzkiBtullamarineB
tuicruisesBtuffersBtuesdaysBtucsonBtubBtuBttwBttheBtsutomiBtsunamisBtsiprasBtshirtsBtshirtBtrynnaBtruthsofBtrustymclustyBtrustingBtrustedBtrulystingsB	truestoryBtruediagnosisB	truckloadB
truckcrashBtruBtrpreston01BtrpBtroyslaby22BtroylercraftBtroyeBtroupeBtroubleonmymindB
trophyhuntBtrophyBtrophiesBtropesBtrooperBtrombonetristanBtrollkrattosBtrollingtilmeekdissBtroisrivieresBtrjdavisBtrixiedrownedBtriviumBtriumphsB
triumphantBtriumphBtrinnaBtrimBtrillacBtriggerBtridentBtricycleBtrickyBtrickxieB	trickshotBtrickierBtriciaoneillphotoBtriciaoneillBtribezBtribeBtribBtriangleBtriadBtreyarchBtreyBtrestleBtrendÛBtrendsBtrendingB
trenchÛBtrenBtremorsBtremontB
tremblayehBtrekkersB	treescapeB	treblinkaBtreatmenB	treasuresBtreasurehouseBtrcBtrayB
travellersBtravelelixirBtrashB
traplord29B
transwomenBtransportationB
transportaB
translatedB
transgressBtransgenderedBtransgenderBtranscriptionB	tramplingBtraitorBtraintragedyBtrainedB
trailheadsBtrailedBtrailBtrafficnetworkBtraditionalistBtraderBtradeBtradcatknightBtracyBtractorB	tracklistBtraceyBtraceBtraBtpsBtprimo24BtoÛBtozletBtoyotaBtoxicsaviorBtoxiccancerdiseasehazardousBtoxicBtownsBtowingBtowerÛªBtowerBtowboatBtowBtoursBtournamentsB
tournamentBtouristsBtoungeBtoughensBtouchingB	touchdownB	tottenhamBtottehamBtotooooooooooB	totooooooBtotalitarianismBtotalitarianBtosuBtossBtosBtoryBtortBtorsoB
torrentialBtorrentB
torrecillaBtorranceB	torontorcBtornadogiveawayB	tormentedBtoriesBtoraBtopicBtopdownBtop25BtootrueBtoosoonBtooooooBtookitlikeamanBtookemB
toocodtoddB
tonysandosBtonymcguinnessB	tonyhsiehBtonycottee1986B	tonyburkeBtonyabbottmhrBtonyBtonneBtonightÛªsBtonguetwisterBtonBtomorrowÛªsBtommorowB	tomlinsonBtomislavBtomfromirelandB	tomdean86B	tomclancyBtolledB	tolewantgB	toleratedB	toleranceB
tokteacherBtoiletsBtoiindianewsBtogtheBtoesBtoenailBtoeBtoddyrockstarBtoddstarnesB
toddcalfeeBtodayÛªsB	todaythatBtodayngrBtodayngBtodayimB	todayhaveB	today4gotBtodBtobiasellwoodBtnwxBtnnBtneazzyBtnaBtnBtms7BtmakeBtlvfacesauspolBtlvfacesBtlozBtlkBtkyonly1fmkBtjrobertson2BtjBtittyBtittieBtitortauBtitoloBtitaniaBtitadomBtitaB
tirelesslyBtireBtipsterBtipBtinybabyBtinyBtintedBtingB	tinderingB	tinderboxBtinderBtimmicallefBtimingB
timeÛBtimesofindiaBtimesapBtimedBtimebombBtimberBtimarobertsBtightlyBtightBtiggrBtigersjostunBtiffanyfrizzellBtierBtideB	tidalhifiBticklemeshawnBtiantaBthÛB	thursdaysBthursdBthursBthurlowBthunderstormtornadoBthundersnowBthuggingBthucydipleaseBthtBthruuuBthrustsBthrowinBthrivingB	threesomeB
threealarmBthreatÛªBthreatintelBthreatconnectBthreadBthrarchivesBthrBthoutaylorbrownB
thoughwillBthouB
thoroughlyBthoriumBthorinsBthorganBthomassmonsonBthomashcrownBthnkBthiÛBthisÛBthisispublichealthBthisisperidotBthisishavehopeB	thisisfazBthisdayinhistoryB
thirtyfiveBthirdquarterBthinnerB	thinkpinkBthingsihateBthhBtheyÛªdBthexfiles201daysBthewesterngazBthetxiBthetshirtkidBthetimepastB	thestrainBthesmallclarkBthesewphistBthesensualeyeBtheresmorewherethatcamefromBthereofBthereisonlysexBthereinB	thereforeBtherealrittzBtherealBtheraminBthepartyofmeannessBthenÛBthenissonianBthenewshypeBtheneedsBthemheBthemermacornBthemeBthemalemadonnaBthemaineBthelegendblueBthejonesesvoiceBthejenmorilloBtheirsB	thehobbitB
thehammersBthegreenpartyBtheghostpartyBtheevilolivesBtheemobragoBtheellenshowBtheeconomistBthedoolinggroupBthedayctBthedarktowerBthedailyshowBthedailybeastBthecomedyquoteBtheburnageblueBthebriankrauseBtheboyofmasksBthebookclubBtheblackshagB
thebargainBthebacheloretteBtheatresBtheatlanticBtheatershootingBtheashesBtheadvocatemagBthdaBthatÛBthatwitchemBthatswhatfriendsareforBthatsabinegirlBthatrussianmanBthatpersianguyBthatnotB
thatfatguyBthatdesBthatdBthankyouBthankuBthankkkBthankingB
thalapathiBthailandBthaiBtfwBtfbBteÛBtextureBtextsBtextingBtexaschainsawmassacreBtexansBtestyBtestifyB	testifiedB	testiclesBtescoBterwilligerB
tersestuffB	territoryB	terrifiedBterrificBterrainB	termn8r13B
terminatedBterellBtepatBtentsBtenshiBtennisBtennewsBtendsBten4BtemÛBtempsB	temporaryBtemporarilyB	templatesBtemperatureBtempBtemeculaBtemecafreemanBtelnetBtellyfckngoB	tellyampiBtellyB	telltalesB
teleportedBtelemarketingBtelekinesisBtelegraphworldB	telanganaBteeÛBteenfictionBteena797BteemoBteeBtedukaBtedcruz2016BtecnoBtechniquB	technicalBtechnewsB
techesbackBteamÛBteamvodgBteamsurvivorsBteamscorpionBteamoB	teammatesBteamhennessyBteamfollowbackBteamatowinnerBteahivetweetsBteafrystlikBtdogBtdmBtcotåÊccotBtcgrenoBtbsBtbrBtblackBtbhBtaylorswift13BtaylorsBtaykreidlerBtayiorrmadeBtaxstoneB	taxreturnB	taxpayersBtaxisBtaxiBtaxesBtawfmcawB
taungbazarBtaufikcjBtattoosBtattooedBtatBtastemycupcakeeBtasksBtaskBtarzanaBtarynelBtarpBtargeB
tareksocalB	taraswartBtapasBtaoistinsightBtantrumsBtanstaafl23BtanslashB
tangletalkBtangledBtanehisicoatesBtamponsBtampabayB
tambourineBtamboBtallestB	talkradioBtalkinghellBtalkedBtalkecologyamphumanBtalismanBtalibansBtalibanBtalesBtakisBtakinBtakeoffBtakehomeBtakecareB	takeawaysBtaipeiBtailorBtailBtahoeblazeravalanches10BtaggingBtaggedBtafsBtaeBtadhgtgmtelBtacticsBtactfulBtacosB	tackettdcBtacitBtaaylordarrBt1000sBsÛªarabiaBszuterBszmnextdoorB
systematicBsysBsyringetoangerBsynapsenkotzeBsymptomsBsymbolBsymantecBsyjexoB
sydtrafficBswtrainsBswornBswordsBswoopingBswollenBswivelsBswitzerlandB	switchingBswitchBswissBswingmanBswimingBswiftycommisshBswiftlyBswellyjetevoBswellBsweetsB	sweetpeasBsweetiebirksBsweepsBsweepingBsweepBswedishBsweatyBsweatfyiBsweatedBsweaBswb1192B
swayoung01BswaybackBswangerBswamiBswagBsvetlanaBsuxBsuvsBsustainourearthBsustainabilityB
suspiciousB	suspendedBsuspectsBsusiyaB
susinessesBsushiB	susanj357B
survivorsrB	survivingBsurveysBsuruÌ¤BsurgicalBsurgesBsurfspaB	surfphotoBsuretyBsureshpprabhuBsureshBsuregodBsupremoBsupremacistB
supposedlyB
supportingB3supporthealthhomebathroomsupportelderlyinjuredsÛB
supportersBsupervBsuperstitionsBsuperstitionB
superpowerBsupernovalesterBsupernaturalBsupermarketBsupermanBsuperiorityBsuperintendentBsuperintendeB	superfoodBsuperbugBsuperbBsunshineBsunsBsunraysB
sunnymeadeB	sunkÛ1B	sunflowerBsundercrBsundayÛªsB
sundaydontBsunburstB	sunburnedBsunbatheBsumoBsumnBsummonsBsummonBsummervibesBsummersBsummerhalleryB
summer2k15BsummaryBsultryBsulBsuitesBsuitedBsuitableBsuingBsuicidesBsuicidebycopBsuhoBsuggsBsugarBsuffieldBsufficientlyBsuffersBsuelinflowerBsuedBsudanÛªsBsuckingBsuckersBsuckedBsuckBsucceedB
subtornadoBsubtletyBsubtleBsubstantialB	substanceB
subsequentBsubsdBsubscriptionBsubmittBsubmissionsB	submergedB	subjectedBsubcontractorBsubconsciousBsubcommitteeB	subatomicBsuBstylistB	stylishlyBstyledBstvmllyBsturyBsturgisBstupidniggrB
stunninglyBstungBstunckleBstumpBstuffinBstudyingB
studebakerBstuddedBstuckinbooksBstuartbroad8B	struttingBstruttedBstruggleBstructuringBstrongmindedBstrivesB
stripteaseBstripsBstrippedBstripeBstripBstrikedBstrictlyB
strickskinB	stretchesBstretcherbearersBstretcherbearerB	stretchedB	stressingBstressesBstrengthBstreetlightBstreetjamzdotnetBstreeBstreamyxhomesouthernBstreamsB	streamingBstrayBstrawberrysoryuBstrawberriesB	stratfordBstrategyhuaB
strategiesBstrapB	strangersBstrandBstraitsBstrainsBstrainB
straightenBstoÛBstowingBstormtrooperB	stormlikeBstormingB	stormfreeB
stormbeardBstoreyBstorenB
stopÛBstoppingBstoponesoundsB
stopharperBstopevictionsBstoodBstonyB	stonewallBstonesBstonebrewingcoBstokesB	stockwellB	stockholmBstoBstlouisBstlndBstlBstirringBstilBstiiiloB	stickynycBstickyBstickingBstickerBsthingBstfxuniversityBstevieBsteveycheese99BstevenrullesBstevenontwatterBstevenBsterotypicalBsternBsterlingscottBsterlingknightBsterlingBstepkansB
stephensonBstephenscifiBstephenkingBstephengeorgBstephaniemarijaB
steph93065BstemmingBstemBstefanoBstefanejonesBsteepB	steellordBstearnsBstealthBsteadyBstdsBstayingBstayedBstavolaB
staverniseBstatisticallyBstationsBstationcdrkellyB
statesÛBstatesvilleB
statementsBstatBstarvingBstarveBstartrekBstartideBstartelegramBstarmadeBstarksBstarkBstarflamegirlBstareBstardateBstarbuckscullyBstarbsBstankyboy88BstandupB
standstillBstandforwolvesBstandardisedBstandardanonymousBstallion150BstalledBstalinsBstalagBstainingBstaidBstagesBstagedB	staffÛBstaffingBstacyBstacksBstackB
stacedemonB	stacdemonBstableBsswBssuBssssnellBssshhheeesshhBsspBssb4BsrslyBsrsBsrkBsriramkB
srajapakseBsr37Bsr22Bsr14BsqwizzixBsquirrelBsquibbyBsqueezedBsqueaverBsqueakyBsquabbleBsqBspyroBspyingBspxB
sputteringB
sputnikintBspursBspurgeonBspurBspså¨BspsgspBspriteB
sprinklersBspringerBspreeBspreadsBspreadBsprayBsprainsBspoutingBspouseBspottingB	spotlightBsportsroadhouseBsportinggoodsBsportenB	spookyfobBsponsorBspongeBspokesBspokeBspokaneBspoiledBspoilBsplifsBsplattershotBsplatoonB	splatlingB	splatdownBsplashBspitsBspitBspiritsBspinsB
spinnelliiBspinBspiltBspillevacuationsredBspikeBspiesB	spiderwebBspicybreadsBspiceBspendsBspencerfearonBspenBspellsBspellB	speedtechBspeedingBspeechBspeculationB
speculatioBspectrumBspecsBspecificallyBspecializedB
specializeBspeccyBspeakingfromexperienceBsparxxxBspartansBsparkzBsparkingBsparkBspanielsBspanielBspanBspammersBspamBspacexBspacewolverineBspaceshiptwoBspaceangelsevenBsoÛBsozBsowB	southÛBsouthwesternB	southwestBsouthridgelifeB	southlineB
southkoreaB	southdownB
southboundBsouthaccidentBsousseBsourmashnumber7BsourB
soundtrackBsoundingBsoundersBsoultechBsoughtBsoudaBsothwestBsosBsorryiBsorrybutitstrueB	sorrowfulBsorrowerBsorrowBsorelyBsophisticationBsophiewiseyBsophieingle01BsoonpandemoniumBsoonergruntB
sonyprousaBsonyBsonoranrattlerBsonofbobbobBsonofbaldwinB	sonisonerBsoniaBsoniB	songhey89BsongforBsonerBsondBsonaBsommeB
sometimesiBsomethinÛªBsomethingyrBsomedayBsolvingB	solelinksBsoleBsoldiBsolanoBsojapanBsoilBsoggyBsoftballBsofaBsodsB
sodamntrueBsocketsBsocketBsockB
socialwotsBsocialtimesBsocialmediadrivenBsociallyBsocalBsocBsobbingB	soapscoopBsoapBsoakerBsoakedBsnuckB
snowywolf5BsnowyBsnowstormhailstormBsnowstormdespiteB	snowflakeB	snotgreenBsnortBsnoopBsnookerBsnippetsBsnipeB
sniiiiiiffBsniffBsniBsneezingBsneaksBsnazzychipzBsnappingBsnapharmonyBsnapchatselfieBsnakesBsnakeBsnackBsnB
smusx16475BsmugglersåÊnabbedB	smugglersBsmugBsmthBsms087809233445BsmsBsmpBsmoresBsmoothedBsmoochyBsmokeyBsmokesBsmokersB
smoakqueenBsmirkingBsmilesBsmfhBsmemB
smelltasteBsmellingBsmelledBsmearedB	smartteksB	smartnewsBsmartBsmantibatamBsmallforestelfBsmallerBsmallbusinessBsmallbizBsmackBslumsBslumberBslspB	slsandpetBslowsBslowpokeB
slosheriffBslosherBslopeofhopeBsloneBsloganBslitBslippingBslipperB
slimebeastB	slightestBslightBslidingB
slideshareBslicedBslewBsleptBsleepjunkiesBslayBslavesBslaveryBslaveB	slaughterBslatukipBslatingBslatedBslashandburnBslappingBslanderBslamsBslammingBslamBslainBslabsBslBskyåÊnewsBskywarsBskywarnBskyscrapersBskyrimBskypeBskynewsBskynetBskylerB
skylandersBskullBskippy6gamingBskipBskinlessBskimsBskimmedBskiingBskiesBskiBskhB
sketchbookBskeletonBskcBskateboardsB	skarletanBskarduBskaggsBsk398BsjubbBsjBsizygwwfBsizewellBsixcarBsivanBsituBsittwayBsiteinvestigatingBsisterÛªBsirtophamhatB
sirtitan45B
sirmixalotBsirmioneBsiriusB
sirenvoiceB
sirensÛBsirensong21B	sirensampBsippinBsipB	siouxlandBsiouxlanBsiouxBsinsBsinkingshipindyBsinkingfundBsinkholeÛB	sinistrasBsingledB	singlecarBsindhB	sincerelyBsince3gBsince1970theB
simulationB
simulatingBsimplifyBsimmonsBsimilarBsilveryB
silverwoodB	silvermanBsilverhuskyBsiloBsillyBsilinskiB
silentmindBsilent0sirisBsilencedBsilasBsikhBsigueBsigninBsignificanceBsignatureschangeBsighBsiftingBsienaBsidjsjdjekdjskdjdBsideÛBsidewalkBsidelinesavageBsidedB
sickÛªBsiblingBshutsB
shunichiroBshuffledBshuffleBshudBshtfBshtapBshtBshrewsBshowwentBshowersstormsBshowersBshowdownBshovelBshoveBshoutoutBshoutedBshououtBshotgunBshortsB
shortfallsBshoppingBshoppeBshootoutåÊB	shootingsBshoookBsholt87BshockingÛÏBshockingBshoalstrafficBshiverBshittonBshiteBshirleyBshipsxanchorsB	shimmyfabBshiiBshiftsBshifterBshiftedBshieldBshiddddBshiasBshiaB
sheÛªsBshevlinhixonBshestooyoungBsherfield72BshenBshemeshBsheltersupportBshellsBshekharguptaBsheetingBshedidBshearBshayolyBshawie17shawieB	shatteredBshatterBsharplyBsharperBsharkÛBsharkBsharifBshariaBsharBshapingBshaperBshapeandBshaolinBshantaeskyyBshantaehalfgenieheroBshantaeforsmashBshanghaiÛªsB	shanaynayBshakjnBshakingcatchingBshakespearesB
shakeologyBshakenBshakeBshaheedBshadowsB	shadowmanBshadowflameBshadowedBshadeBshadBshaBsgc72BsgBsforBsfgiantsBsfaBsexydragonmagicB	sexualityBsexistBsewingBsewardBsewageBseveringBseverelyB	seventiesB	sevenfoldB	sevenfigzBsetting4successBsetsukoBsethalphaeusBsetantaBsessionsBsessionBservingBservicinBservicesft7p7aBsergiopiaggioB	serephinaBsereneBserbianBserasBsequenceBsequalaeBsepticB
separationB	separatedBsenzuBsentientB	sentencedBsensoryBsensorknockB	sensitiveBsenseiB
senschumerB
sensandersBsensBsenfeinsteinBsenatorsBsenatemajldrBsenBseminarsBsemiBsemasirtalksB
selmooooooBselmoBselfseekingBselfpityBselfinflictB
selfesteemBselfdestructionBselfdelusionB
selfavowedBselectsBselectBselBsejorgBseizingBseizeBseismicsoftwareBseismicresistantBsegmentBsegasBsefBseeyouatamicosBseeweedBseemlyBseemethBseekerBseedsBseedBsedarBsedanBsecuringBsecuresB	securedgtBsecuredBsectorsBsectionsB
secondhandB	seclusionBsecBseaworldBseattletimesBseattlesB
seattledotBseatsBseatbeltB
seasonfromBseashoreBseasBseanhannityBseagullsB	seagull07BsdBscynic1BscwxBscumBscufBsct014Bsct012BscseestapreparandoB	scrollingBscriptettesarBscrewedB
screenshotB	screeningB
screechingBscreamsdontB
scratchingB	scratchesBscraptridentBscrapedBscrambledeggsBscoutsBscoutBscourgueBscourgeB	scotto519BscottdpierceBscotrailB
scotiabankBscoredBscorchedBscofieldBscmpnewsBscissorBsciencefictionBscichatBschulzBschoolboyÛªsBscholarsB
schoenfeldBschismÛªBschelbertgeorgBscheerBscenesBscenarioBscegnewsB	scasualtyB	scaryevenBscarletBscariestBscarierBscandalsBscandalBscamBscalpiumB	scaligeroBscabsBsbeeBsaÛB	sayÛBsaynaeBsayinB
sayedridhaBsavsBsavourBsaviorBsavetiBsavesBsaverBsavedenaliwolvesBsavannahross4BsavagesBsavagenationBsaumurBsauldale305BsaudiåÊmosqueBsaudiesBsaudiarabiaB
saturationB	saturatedBsatoshisB
satisfyingBsatireBsatinB
satellitesBsatansBsatanaofhellBsaskBsashaBsarumiBsarniamakchrisB	sarcasticBsarahmcpantsBsarahksilvermanBsaraBsapphirescallopB
sanÛªaBsantosBsantiagoBsantanicopandemoniumB
santaclaraBsansaB	sanonofreBsanityB
sanitizingB	sanitisedBsangBsanfranciscoBsanfordBsanelesstheoryBsandwichBsandunesBsandraBsandersBsandboxBsanctionBsanchezBsamsungBsamsmithworldBsamplesBsampleBsammysositaBsammyBsamihonkonenBsamiB
samelsamelBsamaritansÛªBsamBsalyersblairhallBsalvagesB	salvadorsB
salvadoranBsaluteBsaltyBsaltriverwildhorsesBsaltedBsalopekBsalmanmydarlingBsalmanBsallyB	salisburyBsalesBsaladoBsaladinahmedB
sakuuchihaBsakuBsakhalintribuneBsaisonBsaintsfcBsaintBsailorsBsafyuanBsafsufaBsafferoonicleBsafesB	saferåÊBsafecoBsafarisBsafariBsadtraumatisedB
saddledomeBsaddleB	sacrificeB	sackvilleBsackingsB	sabotageiBsabcnewsroomBsaatBsaalonBsaadtheBs61231aBs5Bs3xleakBs01e09BrzimmermanjrBrytBryrotheunawareBryleedowns02BryansB
ryanoss123BrwrabbitBrvfriedmannB
rvaping101BrvacchianonydnBrvBruthannBrussiaukraineBrussellvilleBrussellB	russaky89BrushlimbaughBruralBrupaulBrunÛBrunninBrunnersB	runkeeperB	runjewelsBruninBrunawayBrunaboutBrumorBrumblingBrumahBrumBrulingBrulerBruledBruhlBrudeB
ruddyyyyyyBrubybotBrubiBrubbingBrubbinBrubberyBruBrtsampdemocracyB
rtrrtcoachB
rtirishirrBrtcomBrsxBrstormcomingBrspcaB	rslm72254BrsfBrsaBrs5B	rs40000crBrrusaBrqBrpnBrpBroyBrowysolouisvilleBrowysoBrowaaBrovingBroutingBroutineBroutesBrouterBrouseyB
roundhouseBround2BrottingBrottentomatoesB	rotationsBrotationBrotatingBrotaryBrostersBrosterBrossmartin7B
rossbartonBroskomnadzorBrosewellBrosenthalauthorB
rosenbergsBrosemarytravaleBrorington95BropesB
roomsgrrrrBroomsBroomrBrooftopsBroofingBroofersBronwydenBronincarbonBrongeBrondarouseyBrondaBronaldBrompBromfordBromesB	romeocrowBromeoBromanticsuspenseBromanticBromaniaBromanatwoodvlogsBromanBromBroloBrollingÛBrolesBroleplayBrolandonabeatsBrokiieeeBroh3smantibatamBroguewatsonBrogersBrogaBrodsBrodkiaiBroddypiperautosB
rodarmer21BrockstarBrocknB
rockinghamBrockingBrochdaleB
robthierenBrobsimssBrobpulsenewsBrobotlvlBrobotcoingameBrobotBrobloxBrobertwelchBrobertoneill31Brobertmeyer9BroberthardingBrobertcaliforniaBrobertbenglundsB
robdelaneyBrobbiewilliamsBrobbedBroarBroanoketimesB	roadworksBroadwaypropertyBroadidBrnkBrlyehBrlauren83199BrjkrrajBrjg0789B
rjailbreakBrizzoBriversBriverroamingB
riveeeeeerBrivalsBritzyjewelsBritualisticBritualBriteBriskyBriserBrisB
ririnsiderBripsB	ripripripBripplesBrippingBriotersBriosladeBriooooosBrio2016B	rinkydnk2BrindouBrinBrijnBrigourBrightlyB	righteousBriggaBrigBrifleBridiculouslyBriddlerBricottaÛBricoBrickybonessxmBricketsBricinBrichhomeydonBrichesBrichelieusaintlaurentBricharkkirkarchBriceechrispiesBriceBribbonBriBrhymesB
rhinestoneBrhiannonBrhettBrhee1975BrgjBrfpB	rfcgeom66BrezaphotographyBreworkedBrewatchingthepilotBrevolutionblightBrevoltBrevitupBreviseB	reviewingBreverseBreversalBrevereBrevengeBrevelBreveillertmB	revealingBreusingBreuniteB	retweetedB	returningBretroactiveBretreatBretractBretooledB
retirementBretireesBretiredfilthBretardB	retainersBresumedBrestrospectBrestoringpathsB	restoringBrestlessnessBrestingBrestartBresqueB
respondingBrespondentsBrespectsB
respectingBresourceBresortBresolvedBresolutevanityBresoluteshieldB	resistantBresinB
resilienceBresigninshameBresidualincomeBresidualBreshrimplevyBreshareworthyBreshapeBresetBreservesBreservedBresemblanceBresearchersBrescuingBrescuerstheBrescuedÛBrescuedagainBrescindB	requiringBrequiemBrequestsBrequaBreqdB
reputationBrepsBreprocussionsBreprisesB
repressionB
representsBrepresentingBrepresentativeBreppedB	reportersBreplacementBreplacedBrepjohnkatkoBrepdonbeyerBrepatriatingBreoccurBrentB
renovationBrenewsitBrenewedBrenew911healthBreneBrenderedBrenderB	renaominoBrenamedBrenB
remymarcelBremoteBremorselessB	remodeledBremixesB	remindersBremindedBremindBremembranceB	remembersBrememberrabaaBremedialBrembrBremasterB
remarkablyBremarkBremandBremainontopB	remainingBremadeBreliveB	religiousB	reliefwebB	relevanceB
relegationB	releasingB	relaxinprB	relationsBrejoiceBrejectdcartoonsB	reiterateBreinedBreinceBreimaginingB	reigncocoBreidlakeBregressBregrBregimeB
reggaeboyzBregentBregcBregBrefusesBrefundsBrefundBrefugeesÛBrefugeesmatterBreflectsBreflectionsB	reflectedBreferredBreferencereferenceB	referenceBreferBreevesBreefBreedBreebokBredwingBreducesBredskinsBredsBredlandsBredistributeBrediscoveredB
rediscoverBredheadB	redhandedBredesigningB
redesignedB
redemptionBredeemerBredeemBreddishBreddevil4lifeBreddakushgoddB	redcliffeBredbullBredbloodBrecruitmentB
recruitingB	recoveredB
recordhighBrecordedB	recordandB	reconnectBrecommendationsBrecoilBrecognitionB
recognisedBrecluseBrecklessBrecipeBrecipBreceivesBrecalledBrecalBrecBreboundBrebootB	rebloggedB
rebelmage2BrebelledBrebeccaBrebahesB
reassignedBrealmBrealliampayneBrealjaxcloneBrealizationsBrealizationB	realitiesB	realisticBrealhotcullenB
realhiphopBreagansBreaganBreafsB	readinessBreaderBreactsBreactorsBreactorbasedB	reactionsBreachedBreaadBrdgB
rdconsiderBrconspiracyB	rcitypornBrchsBrbiBrbcinsuranceBrazedåÊÛÒBrazakBrayquazaerkBraychielovesuBrawfoodblissBravioliåÊwithBraveB	rationingBratioBratingscategoriesBratingsB	ratingbutBrascalBraredealsukBrarB
raptorsbegBrapedBrantsB	rantipoziB	ransackedBranksBrankingBrankedBrankBraniakhalekBrangerkaitimayBrangBrandyBrandomtouristBrandomthoughtBranderson62BrandallpinkstonBramsBrampageBrampBramatBramBralphB	rallyÛBraisinfingersB
raishimi33BrainyBrainwindstormBrainforestresqBraineishidaBrailroadBrailgunsBraidersreporterBrahulkanwalBraheelsharifBraheelBragingBragBraftBraffircBradychildrensBradlerBradiosBradioriffrocksBradicalBrachelcaineBracerBraccoonsBraccoBrabidmonkeys1BrabbitBrabaaB
raabchar28BraBr5liveBr3doBr21Br1354B	qzloremftBqzBquottelevisionBquotoperationsB	quotesttgBquoteofthedayBquoraBquizzedBquitBquirkBquietBquickerBquestsBquestionfatalityflawlessBquestergirlBquemBqueerB
queenwendyBqueenswharfBqueenmyBquarterBquarrelBquantitÛhttpstco64cymg1ltgBqualsBqualitBquakeBquadrillionBqtyBqpr1980BqotringBqnhBqewBqendilBqaveBqampaBq99Bq13Bq1BpythonBpyrotechnicBpyrblissBpyramidhead76BpydisneyBpwhvgwaxBpvrisBputsBputhBpussyxdestroyerBpusssssssssyB	push2leftB	purposelyBpurposeB	purportedBpurpleturtlerdgBpurifiedBpurelyBpurdiesBpuppyshogunBpuppetBpupBpunyBpunkB	punishingB
punishableBpunditsBpunditBpunchBpumpkinBpumperBpumpedBpummelBpulseBpullÛoneÛyouBpullupBpulkovoBpuledotechupdateBpugwashBpugprobsBpugBpuffBpuertoBpuddleBpuckflattenedB
publishingBpublishBpublicityalthoughBpublichealthBpubBpuBpt4Bpt1BpsychrewatchBpsychologistBpsychicBpspBpsfdaBpseudojuuzoBpsdB	psalm3422BpsaBps3Bps2Bps1BprysmianBproxyBproxiesBprovokesB	providersBprovidedBprovenBproveBproudgreenhomeBprotostatesB
protestorsB
protestersBproteinB	protectedBprosserBprosperBprosBproportionsBprophetsBprophecyBpropertycasuBproperlyB	propelledBpropaneB
propagandaBpronouncingBpromsB	promptingBpromptB	promotionBpromotedBpromoBpromisedBpromBprolongBprollyBproliferationBprojectilesÛÒBprogressivesBprogress4ohioBprogramsBprofittothepeopleBprofithungryBprofessionallyBprofbriancoxB
productiveB
productionBproducesBproducerBproduceBproducBprodemocracyBprocBprobsBprobabilityBprobB	privilegeBprisonplanetB	prisonersBprintsBprintingBprintedB	printableB	principleBprincessduckBprinceoffencingBprimalkitchenBprimalBpriestsBpriestBprideBprezBpreviewsBpreventativeB	prevalentBprettyboyshyflizzyB	pretensesBpresumeBprestonÛªsBprestigeBpresstvBpresssecBpressingBpresserBpresleyBpresinkholeBpresidentÛBpresidentialBpresetBpreserveBpresentsBpresentationBpreseasonworkoutsB	preseasonB	preschoolBpresbadBpreppertalkBpreppersBpreparedelectrocutedboilingBprensaBpremisesBpremB
preferableB
prefectureB
preemptiveBpredynasticBpredictionsBpreconditioningBprecisionisticB	precedentB	preachingBpreacherBprayingBprayforsaipanBprayedBpraiseBpragnikBpractitionerB
practicingBpracticenygBpracticallyBprablematiclaBpraBppsellsbabypartsBpporBppleBppfaBppcBppactBpp400drBpp15000266858Bpp15000266818BppBpozeBpozarmyBpoxBpowerwowBpowersBpowerhiroshimaBpowderBpowayBpowBpouringBpouredBpourBpoundsBpoundingBpoundedBpouchBpotterBpotsBpotatoesBpotB	postponedBpostexistenceB	posteringBposterB	postcardsBpostcapitalismBpostalBpossessB	possesionB
positivelyBposesBposBportraitB	portfolioB	portaloosBpornoBpornhubB	porcupineBporciniBpopulation6BpopeyesBpopcornBpop2015BponyBpontingBponeBpollutedBpollsterBpoliticÛBpoliticizedBpoliticiansBpolitelyBpolitB	policylabB
policeÛBpolicengBpoleB	polaroidsBpolarBpolBpokemoncardsBpokemonBpoisonedB	pointÛB	pointlessBpoignantBpogoBpodcastBpodBpoconorecordBpochetteBpocB	pneumoniaBplymouthBplumbingBpluginBplsssBplottedBploppyBpllBplezBpletchÛªsBpledgedBplebBpleasantBpleadedBplazaBplaythroughBplaystationBplayoverwatchB
playingnowBplayaBplattBplatinumBplasticsBplasticBplantcoveredBplantationsBplannersBplannedparenthoodBplankBplaningB	planetaryBplBpkwyBpjcoyleBpizzasBpizzarevBpixelsmovieBpixeljanoszBpixelcanuckBpixarBpivotBpityB
pittsburghBpitmixBpitcherBpitchedBpitBpissBpiscoBpiratesBpirateBpiracyBpirBpiprhysBpipingBpiperwearsthepantsB
pipelinersBpioneerBpineviewBpin23928835BpillsBpilgrimsBpileupBpilesBpileqBpikinBpikachuBpigeonBpigaB	piercingsBpiercingBpierceBpiercB	pieceÛB	pieceofmeBpictwittercompnpizodyBpicturedBpicthisB
pickpocketBpickleBpickensB
pianohandsBpianoB	physicianBphuketBphotoopBphotographsBphotographedB
photogenicBphnotfBphillipsBphillipBphilippinesÛB
philippineBphilippiBphilipduncanBphilipB
phelimkineBpharrellBpharmaB
phantasmalBphandomB
phalaborwaBph0tosBpgaBpftBpfftBpfannebeckersBpettyBpettingBpetitiontakeB
petitionnoB	petersensB
petersburgB	peterknoxBpeterhowenecnBpetereallenBpeterduttonmpBpetelmcguireB	petebestsBpeteBpetcharyBpetaBpetBpestleBperspectivesBperspectiveB	personnelBpersonalizeBpersonalinjuryB
persistentBpersistB	perrychatBperrybellegardeBperrieBperpetratorsB
permissionB
periwinkleBperishedB	periscopeBperformB
perforatedB	perfectlyBperceiveB	pepperoniBpeoplegtBpeoplecommunicationBpensionBpennyBpennliveBpenniesBpenneysB	peninsulaB	penetrateBpenaltyBpelosisBpeiceBpeetersBpeersBpeepedBpeelBpeekedBpeeBpedro20B	pediatricBpedalsBpeasantsBpearlharborBpearlBpealeBpeaceÛªB	peacetimeBpdxabqBpdBpciBpcaldicott7BpbxBpbsBpbohannaBpbcanpcxBpbBpaypileBpaymentBpayingBpaydayprisonBpaydayBpaybackBpaxtonBpawsoxBpawsB
paulstaubsBpaulsBpaulistaBpaulhollywoodB	pattyds50BpatternsBpatrolBpatrickwslsBpatrickjbutlerBpatriciatrainaBpatnaBpatioBpatientreportedBpathsBpathfindersBpatBpasturesBpastorBpastieBpassiveBpassionBpascoeBpascalBpartysBpartnersBpartnerBpartiesBparticipatingBparticipateB	partiallyBparterBpartakeB
parsholicsBparsB	parlimentBparliamentaryBparleyÛªsBparksboardfactsBparksBparkedBparkchatBparisianBparentalBparentBpardonBparchedBparatroopersBparamoreB
paramedicsBparaguayBparadiseBparacordBparaBpapicongressBpapiB	paperworkB	paperbackB	papcrdollBpapaBpantofelBpantiesBpantherBpantalonesfuegoBpanikBpanicsBpanickedBpandoraBpandemoniumisoBpandemicBpancakesBpanamaBpamperedBpampalmaterBpalmoilBpalmerBpalmB	palinfoenBpalestinianÛBpalermoBpalefaceBpaleBpaktheyB	pakistansBpajamasBpaintsBpaintheyBpainfulBpaineBpagingBpageshiBpagesBpaedsBpadsBpadresBpaddytomlinson1
��
Const_5Const*
_output_shapes	
:�N*
dtype0	*��
value��B��	�N"��                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �       	      	      	      	      	      	      	      	      	      		      
	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	       	      !	      "	      #	      $	      %	      &	      '	      (	      )	      *	      +	      ,	      -	      .	      /	      0	      1	      2	      3	      4	      5	      6	      7	      8	      9	      :	      ;	      <	      =	      >	      ?	      @	      A	      B	      C	      D	      E	      F	      G	      H	      I	      J	      K	      L	      M	      N	      O	      P	      Q	      R	      S	      T	      U	      V	      W	      X	      Y	      Z	      [	      \	      ]	      ^	      _	      `	      a	      b	      c	      d	      e	      f	      g	      h	      i	      j	      k	      l	      m	      n	      o	      p	      q	      r	      s	      t	      u	      v	      w	      x	      y	      z	      {	      |	      }	      ~	      	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	      �	       
      
      
      
      
      
      
      
      
      	
      

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
       
      !
      "
      #
      $
      %
      &
      '
      (
      )
      *
      +
      ,
      -
      .
      /
      0
      1
      2
      3
      4
      5
      6
      7
      8
      9
      :
      ;
      <
      =
      >
      ?
      @
      A
      B
      C
      D
      E
      F
      G
      H
      I
      J
      K
      L
      M
      N
      O
      P
      Q
      R
      S
      T
      U
      V
      W
      X
      Y
      Z
      [
      \
      ]
      ^
      _
      `
      a
      b
      c
      d
      e
      f
      g
      h
      i
      j
      k
      l
      m
      n
      o
      p
      q
      r
      s
      t
      u
      v
      w
      x
      y
      z
      {
      |
      }
      ~
      
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
      �
                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �      �                                                                      	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �       �        !      !      !      !      !      !      !      !      !      	!      
!      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !       !      !!      "!      #!      $!      %!      &!      '!      (!      )!      *!      +!      ,!      -!      .!      /!      0!      1!      2!      3!      4!      5!      6!      7!      8!      9!      :!      ;!      <!      =!      >!      ?!      @!      A!      B!      C!      D!      E!      F!      G!      H!      I!      J!      K!      L!      M!      N!      O!      P!      Q!      R!      S!      T!      U!      V!      W!      X!      Y!      Z!      [!      \!      ]!      ^!      _!      `!      a!      b!      c!      d!      e!      f!      g!      h!      i!      j!      k!      l!      m!      n!      o!      p!      q!      r!      s!      t!      u!      v!      w!      x!      y!      z!      {!      |!      }!      ~!      !      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!      �!       "      "      "      "      "      "      "      "      "      	"      
"      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "       "      !"      ""      #"      $"      %"      &"      '"      ("      )"      *"      +"      ,"      -"      ."      /"      0"      1"      2"      3"      4"      5"      6"      7"      8"      9"      :"      ;"      <"      ="      >"      ?"      @"      A"      B"      C"      D"      E"      F"      G"      H"      I"      J"      K"      L"      M"      N"      O"      P"      Q"      R"      S"      T"      U"      V"      W"      X"      Y"      Z"      ["      \"      ]"      ^"      _"      `"      a"      b"      c"      d"      e"      f"      g"      h"      i"      j"      k"      l"      m"      n"      o"      p"      q"      r"      s"      t"      u"      v"      w"      x"      y"      z"      {"      |"      }"      ~"      "      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"      �"       #      #      #      #      #      #      #      #      #      	#      
#      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #       #      !#      "#      ##      $#      %#      &#      '#      (#      )#      *#      +#      ,#      -#      .#      /#      0#      1#      2#      3#      4#      5#      6#      7#      8#      9#      :#      ;#      <#      =#      >#      ?#      @#      A#      B#      C#      D#      E#      F#      G#      H#      I#      J#      K#      L#      M#      N#      O#      P#      Q#      R#      S#      T#      U#      V#      W#      X#      Y#      Z#      [#      \#      ]#      ^#      _#      `#      a#      b#      c#      d#      e#      f#      g#      h#      i#      j#      k#      l#      m#      n#      o#      p#      q#      r#      s#      t#      u#      v#      w#      x#      y#      z#      {#      |#      }#      ~#      #      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#      �#       $      $      $      $      $      $      $      $      $      	$      
$      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $       $      !$      "$      #$      $$      %$      &$      '$      ($      )$      *$      +$      ,$      -$      .$      /$      0$      1$      2$      3$      4$      5$      6$      7$      8$      9$      :$      ;$      <$      =$      >$      ?$      @$      A$      B$      C$      D$      E$      F$      G$      H$      I$      J$      K$      L$      M$      N$      O$      P$      Q$      R$      S$      T$      U$      V$      W$      X$      Y$      Z$      [$      \$      ]$      ^$      _$      `$      a$      b$      c$      d$      e$      f$      g$      h$      i$      j$      k$      l$      m$      n$      o$      p$      q$      r$      s$      t$      u$      v$      w$      x$      y$      z$      {$      |$      }$      ~$      $      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$      �$       %      %      %      %      %      %      %      %      %      	%      
%      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %       %      !%      "%      #%      $%      %%      &%      '%      (%      )%      *%      +%      ,%      -%      .%      /%      0%      1%      2%      3%      4%      5%      6%      7%      8%      9%      :%      ;%      <%      =%      >%      ?%      @%      A%      B%      C%      D%      E%      F%      G%      H%      I%      J%      K%      L%      M%      N%      O%      P%      Q%      R%      S%      T%      U%      V%      W%      X%      Y%      Z%      [%      \%      ]%      ^%      _%      `%      a%      b%      c%      d%      e%      f%      g%      h%      i%      j%      k%      l%      m%      n%      o%      p%      q%      r%      s%      t%      u%      v%      w%      x%      y%      z%      {%      |%      }%      ~%      %      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%      �%       &      &      &      &      &      &      &      &      &      	&      
&      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &       &      !&      "&      #&      $&      %&      &&      '&      (&      )&      *&      +&      ,&      -&      .&      /&      0&      1&      2&      3&      4&      5&      6&      7&      8&      9&      :&      ;&      <&      =&      >&      ?&      @&      A&      B&      C&      D&      E&      F&      G&      H&      I&      J&      K&      L&      M&      N&      O&      P&      Q&      R&      S&      T&      U&      V&      W&      X&      Y&      Z&      [&      \&      ]&      ^&      _&      `&      a&      b&      c&      d&      e&      f&      g&      h&      i&      j&      k&      l&      m&      n&      o&      p&      q&      r&      s&      t&      u&      v&      w&      x&      y&      z&      {&      |&      }&      ~&      &      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&      �&       '      '      '      '      '      '      '      '      '      	'      
'      '      '      '      '      '      
�
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_4Const_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *$
fR
__inference_<lambda>_125837
�
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *$
fR
__inference_<lambda>_125842
8
NoOpNoOp^PartitionedCall^StatefulPartitionedCall
�
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
�

Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*�	
value�	B�	 B�	
�
layer_with_weights-0
layer-0
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	
signatures*
;

	keras_api
_lookup_layer
_adapt_function*
* 
* 
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
trace_2
trace_3* 
6
trace_0
trace_1
trace_2
trace_3* 
* 

serving_default* 
* 
7
	keras_api
lookup_table
token_counts*

trace_0* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
R
_initializer
 _create_resource
!_initialize
"_destroy_resource* 
�
#_create_resource
$_initialize
%_destroy_resource><layer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/*
* 
* 

&trace_0* 

'trace_0* 

(trace_0* 

)trace_0* 

*trace_0* 

+trace_0* 
* 
* 
* 
* 
* 
* 
�
(serving_default_text_vectorization_inputPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCall_1StatefulPartitionedCall(serving_default_text_vectorization_input
hash_tableConstConst_1Const_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_125594
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1Const_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_125879
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameMutableHashTable*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_125892ӱ
�`
�
!__inference__wrapped_model_125321
text_vectorization_input\
Xsequential_2_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle]
Ysequential_2_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	9
5sequential_2_text_vectorization_string_lookup_equal_y<
8sequential_2_text_vectorization_string_lookup_selectv2_t	
identity	��Ksequential_2/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2y
+sequential_2/text_vectorization/StringLowerStringLowertext_vectorization_input*#
_output_shapes
:����������
2sequential_2/text_vectorization/StaticRegexReplaceStaticRegexReplace4sequential_2/text_vectorization/StringLower:output:0*#
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite r
1sequential_2/text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
9sequential_2/text_vectorization/StringSplit/StringSplitV2StringSplitV2;sequential_2/text_vectorization/StaticRegexReplace:output:0:sequential_2/text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
?sequential_2/text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
Asequential_2/text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
Asequential_2/text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
9sequential_2/text_vectorization/StringSplit/strided_sliceStridedSliceCsequential_2/text_vectorization/StringSplit/StringSplitV2:indices:0Hsequential_2/text_vectorization/StringSplit/strided_slice/stack:output:0Jsequential_2/text_vectorization/StringSplit/strided_slice/stack_1:output:0Jsequential_2/text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
Asequential_2/text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Csequential_2/text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Csequential_2/text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
;sequential_2/text_vectorization/StringSplit/strided_slice_1StridedSliceAsequential_2/text_vectorization/StringSplit/StringSplitV2:shape:0Jsequential_2/text_vectorization/StringSplit/strided_slice_1/stack:output:0Lsequential_2/text_vectorization/StringSplit/strided_slice_1/stack_1:output:0Lsequential_2/text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
bsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCastBsequential_2/text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
dsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastDsequential_2/text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
lsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapefsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
lsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
ksequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdusequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0usequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
psequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
nsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatertsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ysequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
ksequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastrsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
nsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
jsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxfsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0wsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
lsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
jsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ssequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0usequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
jsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulosequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0nsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
nsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumhsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0nsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
nsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumhsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0rsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
nsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
osequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountfsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0wsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
isequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumvsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0rsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
msequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
isequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2vsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0jsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0rsequential_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
Ksequential_2/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Xsequential_2_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleBsequential_2/text_vectorization/StringSplit/StringSplitV2:values:0Ysequential_2_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
3sequential_2/text_vectorization/string_lookup/EqualEqualBsequential_2/text_vectorization/StringSplit/StringSplitV2:values:05sequential_2_text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:����������
6sequential_2/text_vectorization/string_lookup/SelectV2SelectV27sequential_2/text_vectorization/string_lookup/Equal:z:08sequential_2_text_vectorization_string_lookup_selectv2_tTsequential_2/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
6sequential_2/text_vectorization/string_lookup/IdentityIdentity?sequential_2/text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������~
<sequential_2/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
4sequential_2/text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������       �
Csequential_2/text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor=sequential_2/text_vectorization/RaggedToTensor/Const:output:0?sequential_2/text_vectorization/string_lookup/Identity:output:0Esequential_2/text_vectorization/RaggedToTensor/default_value:output:0Dsequential_2/text_vectorization/StringSplit/strided_slice_1:output:0Bsequential_2/text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
IdentityIdentityLsequential_2/text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*'
_output_shapes
:����������
NoOpNoOpL^sequential_2/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : : : 2�
Ksequential_2/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Ksequential_2/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:] Y
#
_output_shapes
:���������
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�B
�
__inference_adapt_step_125641
iterator9
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	��IteratorGetNext�(None_lookup_table_find/LookupTableFindV2�,None_lookup_table_insert/LookupTableInsertV2�
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:���������*"
output_shapes
:���������*
output_types
2]
StringLowerStringLowerIteratorGetNext:components:0*#
_output_shapes
:����������
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*#
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite R
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace:output:0StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
UniqueWithCountsUniqueWithCounts"StringSplit/StringSplitV2:values:0*
T0*A
_output_shapes/
-:���������:���������:���������*
out_idx0	�
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:�
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:

_output_shapes
: 
�
�
-__inference_sequential_2_layer_call_fn_125654

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_125376o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�V
�
H__inference_sequential_2_layer_call_and_return_conditional_losses_125579
text_vectorization_inputO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	
identity	��>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2l
text_vectorization/StringLowerStringLowertext_vectorization_input*#
_output_shapes
:����������
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:����������
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������       �
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*'
_output_shapes
:����������
NoOpNoOp?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : : : 2�
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:] Y
#
_output_shapes
:���������
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
__inference__traced_save_125879
file_prefixJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	
savev2_const_6

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�BFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB_CHECKPOINTABLE_OBJECT_GRAPHs
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1savev2_const_6"/device:CPU:0*
_output_shapes
 *
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: ::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
::

_output_shapes
::

_output_shapes
: 
�V
�
H__inference_sequential_2_layer_call_and_return_conditional_losses_125376

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	
identity	��>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Z
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:����������
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:����������
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������       �
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*'
_output_shapes
:����������
NoOpNoOp?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : : : 2�
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
__inference__initializer_1257826
2key_value_init568_lookuptableimportv2_table_handle.
*key_value_init568_lookuptableimportv2_keys0
,key_value_init568_lookuptableimportv2_values	
identity��%key_value_init568/LookupTableImportV2�
%key_value_init568/LookupTableImportV2LookupTableImportV22key_value_init568_lookuptableimportv2_table_handle*key_value_init568_lookuptableimportv2_keys,key_value_init568_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: n
NoOpNoOp&^key_value_init568/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :�N:�N2N
%key_value_init568/LookupTableImportV2%key_value_init568/LookupTableImportV2:!

_output_shapes	
:�N:!

_output_shapes	
:�N
�
�
__inference_<lambda>_1258376
2key_value_init568_lookuptableimportv2_table_handle.
*key_value_init568_lookuptableimportv2_keys0
,key_value_init568_lookuptableimportv2_values	
identity��%key_value_init568/LookupTableImportV2�
%key_value_init568/LookupTableImportV2LookupTableImportV22key_value_init568_lookuptableimportv2_table_handle*key_value_init568_lookuptableimportv2_keys,key_value_init568_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: n
NoOpNoOp&^key_value_init568/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :�N:�N2N
%key_value_init568/LookupTableImportV2%key_value_init568/LookupTableImportV2:!

_output_shapes	
:�N:!

_output_shapes	
:�N
�V
�
H__inference_sequential_2_layer_call_and_return_conditional_losses_125769

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	
identity	��>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Z
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:����������
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:����������
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������       �
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*'
_output_shapes
:����������
NoOpNoOp?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : : : 2�
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
"__inference__traced_restore_125892
file_prefixM
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: 

identity_1��2MutableHashTable_table_restore/LookupTableImportV2�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�BFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB_CHECKPOINTABLE_OBJECT_GRAPHv
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapes
:::*
dtypes
2	�
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:0RestoreV2:tensors:1*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 1
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
IdentityIdentityfile_prefix3^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: S

Identity_1IdentityIdentity:output:0^NoOp_1*
T0*
_output_shapes
: }
NoOp_1NoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: : 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_class
loc:@MutableHashTable
�
�
__inference_restore_fn_125829
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity��2MutableHashTable_table_restore/LookupTableImportV2�
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
�
-
__inference__destroyer_125787
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�V
�
H__inference_sequential_2_layer_call_and_return_conditional_losses_125528
text_vectorization_inputO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	
identity	��>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2l
text_vectorization/StringLowerStringLowertext_vectorization_input*#
_output_shapes
:����������
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:����������
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������       �
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*'
_output_shapes
:����������
NoOpNoOp?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : : : 2�
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:] Y
#
_output_shapes
:���������
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
-
__inference__destroyer_125802
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
;
__inference__creator_125774
identity��
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name569*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
�
�
__inference_save_fn_125821
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	��?MutableHashTable_lookup_table_export_values/LookupTableExportV2�
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::P
add/yConst*
_output_shapes
: *
dtype0*
valueB B
table-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: T
add_1/yConst*
_output_shapes
: *
dtype0*
valueB Btable-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: �

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: �

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:�
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2�
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
�
+
__inference_<lambda>_125842
identityJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�V
�
H__inference_sequential_2_layer_call_and_return_conditional_losses_125453

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	
identity	��>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Z
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:����������
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:����������
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������       �
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*'
_output_shapes
:����������
NoOpNoOp?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : : : 2�
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
G
__inference__creator_125792
identity: ��MutableHashTable|
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_7*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
�
�
-__inference_sequential_2_layer_call_fn_125667

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_125453o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�V
�
H__inference_sequential_2_layer_call_and_return_conditional_losses_125718

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	
identity	��>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Z
text_vectorization/StringLowerStringLowerinputs*#
_output_shapes
:����������
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:���������*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:����������
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������       �
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*'
_output_shapes
:����������
NoOpNoOp?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : : : 2�
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
/
__inference__initializer_125797
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
�
-__inference_sequential_2_layer_call_fn_125387
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_125376o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
#
_output_shapes
:���������
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
$__inference_signature_wrapper_125594
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_125321o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
#
_output_shapes
:���������
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_sequential_2_layer_call_fn_125477
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_125453o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
#
_output_shapes
:���������
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
Y
text_vectorization_input=
*serving_default_text_vectorization_input:0���������H
text_vectorization2
StatefulPartitionedCall_1:0	���������tensorflow/serving/predict:�W
�
layer_with_weights-0
layer-0
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	
signatures"
_tf_keras_sequential
P

	keras_api
_lookup_layer
_adapt_function"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
trace_1
trace_2
trace_32�
-__inference_sequential_2_layer_call_fn_125387
-__inference_sequential_2_layer_call_fn_125654
-__inference_sequential_2_layer_call_fn_125667
-__inference_sequential_2_layer_call_fn_125477�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 ztrace_0ztrace_1ztrace_2ztrace_3
�
trace_0
trace_1
trace_2
trace_32�
H__inference_sequential_2_layer_call_and_return_conditional_losses_125718
H__inference_sequential_2_layer_call_and_return_conditional_losses_125769
H__inference_sequential_2_layer_call_and_return_conditional_losses_125528
H__inference_sequential_2_layer_call_and_return_conditional_losses_125579�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 ztrace_0ztrace_1ztrace_2ztrace_3
�B�
!__inference__wrapped_model_125321text_vectorization_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
serving_default"
signature_map
"
_generic_user_object
L
	keras_api
lookup_table
token_counts"
_tf_keras_layer
�
trace_02�
__inference_adapt_step_125641�
���
FullArgSpec
args�

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_sequential_2_layer_call_fn_125387text_vectorization_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
-__inference_sequential_2_layer_call_fn_125654inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
-__inference_sequential_2_layer_call_fn_125667inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
-__inference_sequential_2_layer_call_fn_125477text_vectorization_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
H__inference_sequential_2_layer_call_and_return_conditional_losses_125718inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
H__inference_sequential_2_layer_call_and_return_conditional_losses_125769inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
H__inference_sequential_2_layer_call_and_return_conditional_losses_125528text_vectorization_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
H__inference_sequential_2_layer_call_and_return_conditional_losses_125579text_vectorization_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
$__inference_signature_wrapper_125594text_vectorization_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
f
_initializer
 _create_resource
!_initialize
"_destroy_resourceR jtf.StaticHashTable
J
#_create_resource
$_initialize
%_destroy_resourceR Z
 ,-
�B�
__inference_adapt_step_125641iterator"�
���
FullArgSpec
args�

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
�
&trace_02�
__inference__creator_125774�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z&trace_0
�
'trace_02�
__inference__initializer_125782�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z'trace_0
�
(trace_02�
__inference__destroyer_125787�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z(trace_0
�
)trace_02�
__inference__creator_125792�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z)trace_0
�
*trace_02�
__inference__initializer_125797�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z*trace_0
�
+trace_02�
__inference__destroyer_125802�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z+trace_0
�B�
__inference__creator_125774"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference__initializer_125782"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference__destroyer_125787"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference__creator_125792"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference__initializer_125797"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference__destroyer_125802"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_save_fn_125821checkpoint_key"�
���
FullArgSpec
args�
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�	
� 
�B�
__inference_restore_fn_125829restored_tensors_0restored_tensors_1"�
���
FullArgSpec
args� 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�
	�	
J
Constjtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant7
__inference__creator_125774�

� 
� "� 7
__inference__creator_125792�

� 
� "� 9
__inference__destroyer_125787�

� 
� "� 9
__inference__destroyer_125802�

� 
� "� @
__inference__initializer_12578223�

� 
� "� ;
__inference__initializer_125797�

� 
� "� �
!__inference__wrapped_model_125321�./0=�:
3�0
.�+
text_vectorization_input���������
� "G�D
B
text_vectorization,�)
text_vectorization���������	j
__inference_adapt_step_125641I1?�<
5�2
0�-�
����������IteratorSpec 
� "
 z
__inference_restore_fn_125829YK�H
A�>
�
restored_tensors_0
�
restored_tensors_1	
� "� �
__inference_save_fn_125821�&�#
�
�
checkpoint_key 
� "���
`�]

name�
0/name 
#

slice_spec�
0/slice_spec 

tensor�
0/tensor
`�]

name�
1/name 
#

slice_spec�
1/slice_spec 

tensor�
1/tensor	�
H__inference_sequential_2_layer_call_and_return_conditional_losses_125528t./0E�B
;�8
.�+
text_vectorization_input���������
p 

 
� "%�"
�
0���������	
� �
H__inference_sequential_2_layer_call_and_return_conditional_losses_125579t./0E�B
;�8
.�+
text_vectorization_input���������
p

 
� "%�"
�
0���������	
� �
H__inference_sequential_2_layer_call_and_return_conditional_losses_125718b./03�0
)�&
�
inputs���������
p 

 
� "%�"
�
0���������	
� �
H__inference_sequential_2_layer_call_and_return_conditional_losses_125769b./03�0
)�&
�
inputs���������
p

 
� "%�"
�
0���������	
� �
-__inference_sequential_2_layer_call_fn_125387g./0E�B
;�8
.�+
text_vectorization_input���������
p 

 
� "����������	�
-__inference_sequential_2_layer_call_fn_125477g./0E�B
;�8
.�+
text_vectorization_input���������
p

 
� "����������	�
-__inference_sequential_2_layer_call_fn_125654U./03�0
)�&
�
inputs���������
p 

 
� "����������	�
-__inference_sequential_2_layer_call_fn_125667U./03�0
)�&
�
inputs���������
p

 
� "����������	�
$__inference_signature_wrapper_125594�./0Y�V
� 
O�L
J
text_vectorization_input.�+
text_vectorization_input���������"G�D
B
text_vectorization,�)
text_vectorization���������	