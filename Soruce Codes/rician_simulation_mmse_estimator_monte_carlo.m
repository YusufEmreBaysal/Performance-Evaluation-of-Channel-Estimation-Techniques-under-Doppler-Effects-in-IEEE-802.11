clear;
clc;
close all;

%% Initial Parameters

add_awgn = true;                    % Add AWGN?
add_rician_fading = true;           % Rician Channel?
simple_rician = true;               % Use Simple Rician Channel

snr_level = 20;                     % SNR Level in dB
doppler_vector = 0:50:5000;         % Doppler value in Hz

payload_length = 100;               % Packet payload size (MSDU size)
monte_carlo_size = 100;               % Monte Carlo Iteration Size
scale_value = 0.2;                  % Image compression scale
modcod = 4;                         % Modulation-Coding Rate (4:16QAM, 1/2 (see wlanNonHTConfig help))

display_spectrum_analyzer = false;      % Display Spectrum Analyzer
display_time_scope = false;             % Display Time Scope
display_constellation_diagram = false;  % Display Constellation Diagram
display_tx_rx_image = false;            % Display TX and RX Image
display_racian_channel = false;         % Display Rician Channel Impulse and frequency responses


%% Rician Channel Settings

if ~simple_rician % çok bozulma
    r_sampleRate = 20e6;
    r_pathDelays = [0 0.5e-6 1.2e-6];
    r_averagePathGains = [0 -3 -6];
    r_KFactor = 10;
    % r_directPathDopplerShift = doppler_value;
    % r_maximumDopplerShift = doppler_value;
    r_dopplerSpectrum = doppler('Bell', 8);
    r_channelFiltering = true;
    r_pathGainsOutputPort = true;
    if display_racian_channel
        r_visualization = 'Impulse and frequency responses';
    else
        r_visualization = 'Off';
    end
end

if simple_rician %% az bozulma
    r_sampleRate = 20e6;
    r_pathDelays = [0 0.1e-6 0.3e-6];
    r_averagePathGains = [0 -1 -2];
    r_KFactor = 30;
    % r_directPathDopplerShift = doppler_value;
    % r_maximumDopplerShift = doppler_value;
    r_dopplerSpectrum = doppler('Bell', 8);
    r_channelFiltering = true;
    r_pathGainsOutputPort = true;
    if display_racian_channel
        r_visualization = 'Impulse and frequency responses';
    else
        r_visualization = 'Off';
    end
end

%% TRANSMITTER
% -------------------------------------------------------------------------

%Configure all the scopes and figures for the example.

% Setup handle for image plot
if ~exist('imFig','var') || ~ishandle(imFig) %#ok<SUSENS>
    imFig = figure;
    imFig.NumberTitle = 'off';
    imFig.Name = 'Image Plot';
    imFig.Visible = 'off';
else
    clf(imFig); % Clear figure
    imFig.Visible = 'off';
end

% Setup Spectrum viewer
spectrumScope = spectrumAnalyzer( ...
    SpectrumType='power-density', ...
    Title='Received Baseband WLAN Signal Spectrum', ...
    YLabel='Power spectral density', ...
    Position=[69 376 800 450]);

% Setup the constellation diagram viewer for equalized WLAN symbols
refQAM = wlanReferenceSymbols('16QAM');
constellation = comm.ConstellationDiagram(...
    Title='Equalized WLAN Symbols',...
    ShowReferenceConstellation=true,...
    ReferenceConstellation=refQAM,...
    Position=[878 376 460 460]);

%% Prepare Image File

% Input an image file and convert to binary stream
fileTx = 'peppers.png';                          % Image file name
fData = imread(fileTx);                          % Read image data from file
scale = scale_value;                                     % Image scaling factor
origSize = size(fData);                          % Original input image size
scaledSize = max(floor(scale.*origSize(1:2)),1); % Calculate new image size
heightIx = min(round(((1:scaledSize(1))-0.5)./scale+0.5),origSize(1));
widthIx = min(round(((1:scaledSize(2))-0.5)./scale+0.5),origSize(2));
fData = fData(heightIx,widthIx,:);               % Resize image
imsize = size(fData);                            % Store new image size
txImage = fData(:);

if (display_tx_rx_image)
    % Plot transmit image
    imFig.Visible = 'on';
    subplot(211);
    imshow(fData);
    title('Transmitted Image');
    subplot(212);
    title('Received image appears here...');
    set(gca,'Visible','off');
end

set(findall(gca, 'type', 'text'), 'visible', 'on');

%% Fragment Transmit Data

%msduLength = 2304; % MSDU length in bytes
msduLength = payload_length; % MSDU length in bytes
numMSDUs = ceil(length(txImage)/msduLength);
padZeros = msduLength - mod(length(txImage),msduLength);
txData = [txImage;zeros(padZeros,1)];
txDataBits = double(reshape(de2bi(txData, 8)',[],1));

% Divide input data stream into fragments
bitsPerOctet = 8;
data = zeros(0,1);

for i=0:numMSDUs-1

    % Extract image data (in octets) for each MPDU
    frameBody = txData(i*msduLength+1:msduLength*(i+1),:);

    % Create MAC frame configuration object and configure sequence number
    cfgMAC = wlanMACFrameConfig(FrameType='Data',SequenceNumber=i);

    % Generate MPDU
    [psdu, lengthMPDU]= wlanMACFrame(frameBody,cfgMAC,OutputFormat='bits');

    % Concatenate PSDUs for waveform generation
    data = [data; psdu]; %#ok<AGROW>

end

fprintf("SIMULATION SETTINGS:\n");
fprintf("======================================================================\n");

fprintf("Add AWGN                     : %s\n", string(add_awgn));
fprintf("Rician Channel               : %s\n", string(add_rician_fading));
fprintf("Simple Rician Channel        : %s\n", string(simple_rician));
fprintf("SNR Level (dB)               : %d\n", snr_level);
fprintf("Doppler Vector (Hz)          : %d:%d:%d\n", doppler_vector(1), diff(doppler_vector(1:2)), doppler_vector(end));
fprintf("Simulation Doppler Count     : %d\n", length(doppler_vector));
fprintf("Payload Length (Bytes, MSDU) : %d\n", payload_length);
fprintf("MPDU Length                  : %d\n", lengthMPDU);
fprintf("Monte Carlo Size             : %d\n", monte_carlo_size);
fprintf("Image Scale Value            : %.2f\n", scale_value);
fprintf("Modulation-Coding Rate       : %d (4:16QAM, 1/2)\n", modcod);
fprintf("Total Step                   : %d\n", length(doppler_vector) * monte_carlo_size);
fprintf("Packet Count (num of MSDU)   : %d\n", numMSDUs);

fprintf("======================================================================\n");
fprintf("DISPLAY SETTINGS:\n");
fprintf("Spectrum Analyzer            : %s\n", string(display_spectrum_analyzer));
fprintf("Time Scope                   : %s\n", string(display_time_scope));
fprintf("Constellation Diagram        : %s\n", string(display_constellation_diagram));
fprintf("TX and RX Image              : %s\n", string(display_tx_rx_image));
fprintf("Rician Channel Display       : %s\n", string(display_racian_channel));
fprintf("======================================================================\n");

fprintf("RICIAN CHANNEL DETAILS:\n");
fprintf("Sample Rate                  : %.1e Hz\n", r_sampleRate);
fprintf("Path Delays (s)              : %s\n", mat2str(r_pathDelays));
fprintf("Average Path Gains (dB)      : %s\n", mat2str(r_averagePathGains));
fprintf("K-Factor                     : %d\n", r_KFactor);
fprintf("Channel Filtering            : %s\n", string(r_channelFiltering));
fprintf("Path Gains Output Port       : %s\n", string(r_pathGainsOutputPort));
fprintf("Visualization                : %s\n", r_visualization);
fprintf("======================================================================\n");

fprintf("\nSIMULATIONS\n");

%% Generate 802.11a Baseband WLAN Signal

nonHTcfg = wlanNonHTConfig;       % Create packet configuration
nonHTcfg.MCS = modcod;                 % Modulation: 64QAM Rate: 2/3
nonHTcfg.NumTransmitAntennas = 1; % Number of transmit antenna
chanBW = nonHTcfg.ChannelBandwidth;
nonHTcfg.PSDULength = lengthMPDU; % Set the PSDU length

scramblerInitialization = randi([1 127],numMSDUs,1);

osf = 1.5;

sampleRate = wlanSampleRate(nonHTcfg); % Nominal sample rate in Hz

% Generate baseband NonHT packets separated by idle time
txWaveform = wlanWaveformGenerator(data,nonHTcfg, ...
    NumPackets=numMSDUs,IdleTime=20e-6, ...
    ScramblerInitialization=scramblerInitialization,...
    OversamplingFactor=osf);

%% Loops: Doppler and Monte Carlo
% =========================================================================
% The entire process will be repeated monte_carlo_size times for each doppler value.

totalDopplerCount = length(doppler_vector);
totalSimCount = totalDopplerCount * monte_carlo_size;

simCounter = 0;  % iteration counter

ber_results = zeros(totalDopplerCount, monte_carlo_size);

for dIdx = 1:totalDopplerCount

    currentDoppler = doppler_vector(dIdx);

    for mcIdx = 1:monte_carlo_size
        tic
        simCounter = simCounter + 1;

        % Recreate Rician Channel object
        ricianChannel = comm.RicianChannel( ...
            SampleRate=r_sampleRate, ...
            PathDelays=r_pathDelays, ...
            AveragePathGains=r_averagePathGains, ...
            KFactor=r_KFactor, ...
            DirectPathDopplerShift=currentDoppler, ...
            MaximumDopplerShift=currentDoppler, ...
            DopplerSpectrum=r_dopplerSpectrum, ...
            ChannelFiltering=r_channelFiltering, ...
            PathGainsOutputPort=r_pathGainsOutputPort, ...
            Visualization=r_visualization);

        % % disp(ricianChannel);

        %% Receiver

        %ricianAddedWaveform;
        if add_rician_fading
            ricianAddedWaveform = ricianChannel(txWaveform);
        else
            ricianAddedWaveform = txWaveform;
        end

        %awgnAddedWaveform;
        if add_awgn == true
            awgnAddedWaveform = awgn(ricianAddedWaveform,snr_level,'measured');   % Sinyal awgn kanalından geçirilir
        else
            awgnAddedWaveform = ricianAddedWaveform;
        end

        rxWaveform = awgnAddedWaveform;


        %% Spectrums

        if (display_time_scope)
            scopes = timescope;
            scopes(txWaveform, rxWaveform );
        end

        if (display_spectrum_analyzer)
            spectrumScope.SampleRate = sampleRate*osf;
            spectrumScope(txWaveform, rxWaveform);
            release(spectrumScope);
        end

        %% Receiver Processing

        aStop = 40;                                             % Stopband attenuation
        ofdmInfo = wlanNonHTOFDMInfo('NonHT-Data',nonHTcfg);    % OFDM parameters
        SCS = sampleRate/ofdmInfo.FFTLength;                    % Subcarrier spacing
        txbw = max(abs(ofdmInfo.ActiveFrequencyIndices))*2*SCS; % Occupied bandwidth
        [L,M] = rat(1/osf);
        maxLM = max([L M]);
        R = (sampleRate-txbw)/sampleRate;
        TW = 2*R/maxLM;                                         % Transition width
        b = designMultirateFIR(L,M,TW,aStop);

        firrc = dsp.FIRRateConverter(L,M,b);
        rxWaveform = firrc(rxWaveform);

        displayFlag = false;

        rxWaveformLen = size(rxWaveform,1);
        searchOffset = 0; % Offset from start of the waveform in samples

        ind = wlanFieldIndices(nonHTcfg);
        Ns = ind.LSIG(2)-ind.LSIG(1)+1; % Number of samples in an OFDM symbol

        % Minimum packet length is 10 OFDM symbols
        lstfLen = double(ind.LSTF(2)); % Number of samples in L-STF
        minPktLen = lstfLen*5;
        pktInd = 1;
        fineTimingOffset = [];
        packetSeq = [];
        rxBit = [];

        % Perform EVM calculation
        evmCalculator = comm.EVM(AveragingDimensions=[1 2 3]);
        evmCalculator.MaximumEVMOutputPort = true;


        iter = 0;
        while (searchOffset+minPktLen)<=rxWaveformLen

            iter = iter + 1;
            % Packet detect
            pktOffset = wlanPacketDetect(rxWaveform,chanBW,searchOffset,0.5);

            % Adjust packet offset
            pktOffset = searchOffset+pktOffset;
            if isempty(pktOffset) || (pktOffset+double(ind.LSIG(2))>rxWaveformLen)
                if pktInd==1
                    %disp('** No packet detected **');
                end
                break;
            end

            nonHT = rxWaveform(pktOffset+(ind.LSTF(1):ind.LSIG(2)),:);
            coarseFreqOffset = wlanCoarseCFOEstimate(nonHT,chanBW);
            nonHT = frequencyOffset(nonHT,sampleRate,-coarseFreqOffset);

            % Symbol timing synchronization
            fineTimingOffset = wlanSymbolTimingEstimate(nonHT,chanBW);

            % Adjust packet offset
            pktOffset = pktOffset+fineTimingOffset;

            % Timing synchronization complete: Packet detected and synchronized
            % Extract the non-HT preamble field after synchronization and
            % perform frequency correction
            if (pktOffset<0) || ((pktOffset+minPktLen)>rxWaveformLen)
                searchOffset = pktOffset+1.5*lstfLen;
                continue;
            end
            % fprintf('\nPacket-%d detected at index %d\n',pktInd,pktOffset+1); % yorum

            % Extract first 7 OFDM symbols worth of data for format detection and
            % L-SIG decoding
            nonHT = rxWaveform(pktOffset+(1:7*Ns),:);
            nonHT = frequencyOffset(nonHT,sampleRate,-coarseFreqOffset);

            % Perform fine frequency offset correction on the synchronized and
            % coarse corrected preamble fields
            lltf = nonHT(ind.LLTF(1):ind.LLTF(2),:);           % Extract L-LTF
            fineFreqOffset = wlanFineCFOEstimate(lltf,chanBW);
            nonHT = frequencyOffset(nonHT,sampleRate,-fineFreqOffset);
            cfoCorrection = coarseFreqOffset+fineFreqOffset; % Total CFO

            % Channel estimation using L-LTF
            lltf = nonHT(ind.LLTF(1):ind.LLTF(2),:);
            demodLLTF = wlanLLTFDemodulate(lltf,chanBW);
            %chanEstLLTF = wlanLLTFChannelEstimate(demodLLTF,chanBW);


            %% MMSE Channel Estimation

            % 1) Perform L-LTF demodulation using the MATLAB function:
            % Dimensions of demodLLTF: Subcarrier x Nsym x Nrx (e.g., 52 active subcarriers)

            % 2) Obtain the L-LTF reference (via MATLAB function or manually):
            % This contains the modulation values for each subcarrier as per the 802.11 standard.
            % The function below returns theoretical (complex) values for 2 L-LTF symbols.
            %ltfRef = wlanLLTF(nonHTcfg);
            % Manually defined L-LTF reference symbols (for 52 active subcarriers)
            ltfRef = [
                0.992998773257078 - 2.10315137112311e-17i;
                1.00603237146065 - 1.09003744987712e-16i;
                -1.00960738068105 + 1.80139217356197e-17i;
                -1.00442540775260 + 3.10761649705418e-16i;
                0.997792490804252 + 8.72594037746279e-17i;
                0.993136288235564 - 1.28472535432169e-16i;
                -0.994092035878726 - 7.32148404718551e-17i;
                0.994614150338827 - 1.18740974729705e-16i;
                -0.997558135445432 - 2.36356103820944e-16i;
                1.00076289062380 + 3.35956023021503e-17i;
                1.00322339085385 - 2.46198353601109e-16i;
                1.00658729405644 - 3.39481316910856e-17i;
                1.00423126134737 - 2.08670546952796e-16i;
                1.00158838890813 + 1.55837152780020e-16i;
                0.996538880233758 - 2.33689084718338e-16i;
                -0.994098308351914 - 8.29813697477657e-18i;
                -0.996830445392550 + 2.04135326178647e-16i;
                0.997272498789695 + 3.16311397026503e-16i;
                0.998632785675455 - 1.52823340164094e-16i;
                -1.00310241074637 - 2.72513246831270e-16i;
                1.00182162395430 + 3.72223282435439e-17i;
                -1.00429506341796 - 2.34617201688488e-16i;
                1.00331001022231 + 6.27600897000523e-20i;
                1.00114237149355 + 7.28223224764710e-17i;
                0.998588310652260 + 1.13147841104664e-17i;
                0.996568399161729 + 4.00343559907704e-17i;
                0.996568399161729 + 1.16163094902167e-16i;
                -0.998588310652260 + 1.91626057295876e-16i;
                -1.00114237149355 - 2.06931644820340e-16i;
                1.00331001022231 + 2.16644595061418e-16i;
                1.00429506341796 - 4.58631816754533e-16i;
                -1.00560068730616 - 1.71142740304483e-16i;
                1.00035779976559 + 4.11074816055405e-17i;
                -0.998632785675455 - 3.50584113925398e-17i;
                0.995946106930726 - 7.84354599010849e-17i;
                -0.993899342564132 - 5.42668179004640e-17i;
                -0.994098308351914 + 1.89961463336788e-16i;
                -0.999349917317429 - 1.43977951309401e-16i;
                -1.00038497379181 + 5.81668023641600e-17i;
                -1.00346141240354 + 2.94367227460650e-16i;
                1.00416232700911 - 2.14075285629261e-16i;
                1.00322339085385 - 1.71142740304483e-16i;
                -1.00363106118831 + 2.66274321495419e-16i;
                -0.999112152605094 - 8.52224482846008e-18i;
                0.994880206477172 - 1.44385300406485e-16i;
                -0.992132387259844 - 1.33614933656170e-16i;
                0.993136288235564 - 8.29813697477657e-18i;
                -1.00070339817140 - 1.96161278070025e-16i;
                1.00629707004272 + 1.60889438399985e-17i;
                1.00960738068105 - 6.43694516171619e-17i;
                1.00754927742129 - 9.56054697374051e-17i;
                0.990302236443024 - 1.04303893431548e-16i
                ];

            % 3) Compute proportional relationship similar to LS estimation:
            %    Dimensions of demodLLTF: [Nsubcarriers x 2 x Nrx]
            %    Dimensions of ltfRef:    [Nsubcarriers x 2]
            lsEst = demodLLTF ./ ltfRef;  % Element-wise division

            % 4) There are 2 symbols (L-LTF1 and L-LTF2). Take the mean to reduce noise:
            lsEst = mean(lsEst, 2);  % Output dimensions => [Nsubcarriers x 1 x Nrx]

            % 5) Noise variance is already being calculated:
            %    noiseVarNonHT = wlanLLTFNoiseEstimate(demodLLTF);
            %    You can reuse the same variable here if it is already computed.

            % 6) Compute MMSE coefficient.
            %    Simplified assumption: Assume the magnitude squared of ltfRef ~ 1 (or constant).
            %    For each subcarrier: H_MMSE = H_LS * ( |X|^2 / (|X|^2 + NoiseVar) )
            %    = H_LS * (1 / (1 + NoiseVar)) [if |X|=1 is assumed].
            %
            %    In a more realistic scenario, ltfRef amplitude can vary per subcarrier.
            %    noiseVarNonHT can also be a scalar or subcarrier-specific.
            %    Here, we show the "simplest" approach:

            % (a) Find |X|^2 (mean power of reference per subcarrier)
            refPower = mean(abs(ltfRef).^2, 2); % [Nsubcarriers x 1]

            % Noise estimation
            %noiseVarNonHT = helperNoiseEstimate(demodLLTF); % old
            noiseVarNonHT = wlanLLTFNoiseEstimate(demodLLTF);

            % (b) MMSE coefficient: alpha = refPower ./ (refPower + noiseVarNonHT)
            %    (If noiseVarNonHT is scalar, it will be broadcast per subcarrier)
            alpha = refPower ./ (refPower + noiseVarNonHT);

            % alpha => [Nsubcarriers x 1], lsEst => [Nsubcarriers x 1 x Nrx]
            % Multiply element-wise via broadcasting:
            mmseEst = lsEst .* alpha;  % Element-wise multiplication => [Nsubcarriers x 1 x Nrx]

            % 7) Adjust dimensions:
            chanEstLLTF = squeeze(mmseEst); % [Nsubcarriers x Nrx]





            % Noise estimation
            %noiseVarNonHT = helperNoiseEstimate(demodLLTF); % old
            noiseVarNonHT = wlanLLTFNoiseEstimate(demodLLTF);

            % Packet format detection using the 3 OFDM symbols immediately
            % following the L-LTF
            format = wlanFormatDetect(nonHT(ind.LLTF(2)+(1:3*Ns),:), ...
                chanEstLLTF,noiseVarNonHT,chanBW);
            %disp(['  ' format ' format detected']);
            if ~strcmp(format,'Non-HT')
                %fprintf('  A format other than Non-HT has been detected\n');
                searchOffset = pktOffset+1.5*lstfLen;
                continue;
            end

            % Recover L-SIG field bits
            [recLSIGBits,failCheck] = wlanLSIGRecover( ...
                nonHT(ind.LSIG(1):ind.LSIG(2),:), ...
                chanEstLLTF,noiseVarNonHT,chanBW);

            if failCheck
                %fprintf('  L-SIG check fail \n');
                searchOffset = pktOffset+1.5*lstfLen;
                continue;
            else
                %fprintf('  L-SIG check pass \n');
            end

            % Retrieve packet parameters based on decoded L-SIG
            [lsigMCS,lsigLen,rxSamples] = helperInterpretLSIG(recLSIGBits,sampleRate);

            if (rxSamples+pktOffset)>length(rxWaveform)
                %disp('** Not enough samples to decode packet **');
                break;
            end

            % Apply CFO correction to the entire packet
            rxWaveform(pktOffset+(1:rxSamples),:) = frequencyOffset(...
                rxWaveform(pktOffset+(1:rxSamples),:),sampleRate,-cfoCorrection);

            % Create a receive Non-HT config object
            rxNonHTcfg = wlanNonHTConfig;
            rxNonHTcfg.MCS = lsigMCS;
            rxNonHTcfg.PSDULength = lsigLen;

            % Get the data field indices within a PPDU
            indNonHTData = wlanFieldIndices(rxNonHTcfg,'NonHT-Data');

            % Recover PSDU bits using transmitted packet parameters and channel
            % estimates from L-LTF
            [rxPSDU,eqSym] = wlanNonHTDataRecover(rxWaveform(pktOffset+...
                (indNonHTData(1):indNonHTData(2)),:), ...
                chanEstLLTF,noiseVarNonHT,rxNonHTcfg);

            % Show current constellation
            if (display_constellation_diagram && mod(iter, floor(numMSDUs/10)) == 0)
                constellation(reshape(eqSym,[],1));
                release(constellation);
            end

            refSym = wlanClosestReferenceSymbol(eqSym,rxNonHTcfg);
            [evm.RMS,evm.Peak] = evmCalculator(refSym,eqSym);

            % Decode the MPDU and extract MSDU
            [cfgMACRx,msduList{pktInd},status] = wlanMPDUDecode(rxPSDU,rxNonHTcfg); %#ok<*SAGROW>

            if strcmp(status,'Success')
                %disp('  MAC FCS check pass');

                % Store sequencing information
                packetSeq(pktInd) = cfgMACRx.SequenceNumber;

                % Convert MSDU to a binary data stream
                %rxBit{pktInd} = reshape(de2bi(hex2dec(cell2mat(msduList{pktInd})),8)',[],1); % old
                rxBit{pktInd} = int2bit(hex2dec(cell2mat(msduList{pktInd})),8,false);

            else % Decoding failed
                if strcmp(status,'FCSFailed')
                    % FCS failed
                    %disp('  MAC FCS check fail');
                else
                    % FCS passed but encountered other decoding failures
                    %disp('  MAC FCS check pass');
                end

                % Since there are no retransmissions modeled in this example, we
                % extract the image data (MSDU) and sequence number from the MPDU,
                % even though FCS check fails.

                % Remove header and FCS. Extract the MSDU.
                macHeaderBitsLength = 24*bitsPerOctet;
                fcsBitsLength = 4*bitsPerOctet;
                msduList{pktInd} = rxPSDU(macHeaderBitsLength+1:end-fcsBitsLength);

                % Extract and store sequence number
                sequenceNumStartIndex = 23*bitsPerOctet+1;
                sequenceNumEndIndex = 25*bitsPerOctet-4;
                % packetSeq(pktInd) = bi2de(rxPSDU(sequenceNumStartIndex:sequenceNumEndIndex)'); % old
                conversionLength = sequenceNumEndIndex - sequenceNumStartIndex + 1;
                packetSeq(pktInd) = bit2int(rxPSDU(sequenceNumStartIndex:sequenceNumEndIndex),conversionLength,false);

                % MSDU binary data stream
                rxBit{pktInd} = double(msduList{pktInd});
            end

            % Display decoded information
            if displayFlag
                %fprintf('  Estimated CFO: %5.1f Hz\n\n',cfoCorrection); %#ok<*UNRCH>

                %disp('  Decoded L-SIG contents: ');
                %fprintf('                            MCS: %d\n',lsigMCS);
                %fprintf('                         Length: %d\n',lsigLen);
                %fprintf('    Number of samples in packet: %d\n\n',rxSamples);

                %fprintf('  EVM:\n');
                %fprintf('    EVM peak: %0.3f%%  EVM RMS: %0.3f%%\n\n', ...
                %    evm.Peak,evm.RMS);

                %fprintf('  Decoded MAC Sequence Control field contents:\n');
                %fprintf('    Sequence number: %d\n\n',packetSeq(pktInd));
            end

            % Update search index
            searchOffset = pktOffset+double(indNonHTData(2));

            % Finish processing when a duplicate packet is detected. The
            % recovered data includes bits from duplicate frame
            % Remove the data bits from the duplicate frame
            % if length(unique(packetSeq)) < length(packetSeq)
            %     rxBit = rxBit(1:length(unique(packetSeq)));
            %     packetSeq = packetSeq(1:length(unique(packetSeq)));
            %     continue
            % end

            pktInd = pktInd+1;

        end % while (searchOffset+minPktLen)<=rxWaveformLen

        % Show final constellation
        if (display_constellation_diagram)
            constellation(reshape(eqSym,[],1)); % Current constellation
        end

        %%

        if ~(isempty(fineTimingOffset) || isempty(pktOffset))

            rxData = cell2mat(rxBit);

            % Remove duplicate packets if any. Duplicate packets are located at the
            % end of rxData
            if length(packetSeq)>numMSDUs
                numDupPackets = size(rxData,2)-numMSDUs;
                rxData = rxData(:,1:end-numDupPackets);
            end

            % Initialize variables for while loop
            startSeq = [];
            i=-1;

            % Only execute this if one of the packet sequence values have been decoded
            % accurately
            %if any(packetSeq<numMSDUs)
            if true
                while isempty(startSeq)
                    % This searches for a known packetSeq value
                    i = i + 1;
                    startSeq = find(packetSeq==i);
                end
                % Circularly shift data so that received packets are in order for image reconstruction. It
                % is assumed that all packets following the starting packet are received in
                % order as this is how the image is transmitted.
                rxData = circshift(rxData,[0 -(startSeq(1)-i-1)]); % Order MAC fragments

                % Perform bit error rate (BER) calculation on reordered data
                bitErrorRate = comm.ErrorRate;
                err = bitErrorRate(double(rxData(:)), ...
                    txDataBits(1:length(reshape(rxData,[],1))));

                %fprintf('  \nBit Error Rate (BER):\n');
                %fprintf('          Bit Error Rate (BER) = %0.5f\n',err(1));
                %fprintf('          Number of bit errors = %d\n',err(2));
                %fprintf('    Number of transmitted bits = %d\n\n',length(txDataBits));

                %% finalBer
                finalBer = err(1);

            end

            decData = bi2de(reshape(rxData(:),8,[])');

            % Append NaNs to fill any missing image data
            if length(decData)<length(txImage)
                numMissingData = length(txImage)-length(decData);
                decData = [decData;NaN(numMissingData,1)];
            else
                decData = decData(1:length(txImage));
            end

            if (display_tx_rx_image)

                % Recreate image from received data
                fprintf('\nConstructing image from received data.\n');
                receivedImage = uint8(reshape(decData,imsize));

                % Plot received image
                if exist('imFig','var') && ishandle(imFig) % If Tx figure is open
                    figure(imFig); subplot(212);
                else
                    figure; subplot(212);
                end

                imshow(receivedImage);
                title(sprintf('Received Image'));

            end

        else
            % Herhangi bir paket bulunamadıysa BER = 1
            finalBer = 0.5;
        end

        %% ber_results
        if ~exist('finalBer','var')
            % ForSafety
            finalBer = 0.5;
        end
        ber_results(dIdx, mcIdx) = finalBer;


        %calculate remeainig time
        delta_time(simCounter) = toc;

        % Calculate the moving average for the last 100 values
        if length(delta_time) >= 100
            avg_delta_time = mean(delta_time(end-99:end)); % Last 100 values
        else
            avg_delta_time = mean(delta_time); % If less than 100 values, use all
        end

        avg_remaining_time = ceil(avg_delta_time * totalSimCount - avg_delta_time * simCounter);
        avg_remaining_time_min = ceil(avg_remaining_time/60); 

        % Display Results
        % Process = xx/100  (total work percentage)
        currentProgress = (simCounter / totalSimCount)*100;
        fprintf('Step                         : %d/%d\n', simCounter, totalSimCount);
        fprintf('Process                      : %.2f/100%%\n', currentProgress);
        fprintf('Remaining Time               : %d min\n', avg_remaining_time_min);
        fprintf('Average Simulation Speed     : %.2f seconds per step\n', avg_delta_time);
        fprintf('Doppler                      : %d Hz\n', currentDoppler);
        fprintf('Monte Carlo                  : %d/%d\n', mcIdx, monte_carlo_size);
        fprintf('BER                          : %.3f\n', finalBer);
        fprintf('Av. BER for %d Hz doppler  : %.3f\n\n',currentDoppler, mean(ber_results(dIdx, 1:mcIdx)));

    end % monte_carlo_size

end % doppler_vector

%% Monte Carlo Average and Graphs
average_ber = mean(ber_results, 2); 


figure
% Interpolation
doppler_interp = doppler_vector(1):1:doppler_vector(end);
ber_interp = interp1(doppler_vector, average_ber*100, doppler_interp, 'spline'); % Using spline interpolation

plot(doppler_interp, ber_interp, '-', "LineWidth", 4, 'Color', [0.8500 0.3250 0.0980]);
hold on;
plot(doppler_vector, average_ber*100, '*',"LineWidth", 2, 'Color', [0 0.4470 0.7410]);

xlabel('Doppler (Hz)');
ylabel("BER(%)")
title('MMSE Estimator Doppler - BER Graph');
grid on;
hold off;

% Curve Fitting
% Perform Curve Fitting
fit_type = 'smoothingspline'; % You can change this to other fit types, e.g., 'poly2', 'poly3', etc.
fitted_curve = fit(doppler_vector(:), (average_ber' * 100)', fit_type);

% Generate fine-grained data for plotting the fitted curve
doppler_fine = 0:1:5000;
ber_fit = fitted_curve(doppler_fine);

% Plot the data and fitted curve
figure;
plot(doppler_fine, ber_fit, '-', "LineWidth", 4, 'Color', [0.8500 0.3250 0.0980], 'DisplayName', 'Fitted Curve'); % Fitted curve
hold on;

plot(doppler_vector, average_ber * 100, '*', "LineWidth", 2, 'Color', [0 0.4470 0.7410], 'DisplayName', 'Original Data'); % Original data points
xlabel('Doppler (Hz)');
ylabel("BER (%)");
title('MMSE Estimator Doppler - BER Graph');
legend show;
grid on;
hold off;
