import{_ as s,c as a,o as n,a4 as i}from"./chunks/framework.ClmyTGR2.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, AMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">      Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the spirals</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [MLUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Datasets</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">make_spiral</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(sequence_length) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dataset_size]</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the labels</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    clockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sequence_length], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                         for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    anticlockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                                 d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, (sequence_length </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                             for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[((dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Float32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(clockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, anticlockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; dims</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Split the dataset</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (x_train, y_train), (x_val, y_val) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> splitobs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_data, labels); at</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create DataLoaders</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Use DataLoader to automatically minibatch and shuffle the data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_train, y_train)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Don&#39;t shuffle the validation data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_val, y_val)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR832/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR832/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR832/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Main.var&quot;##225&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # First we will have to run the sequence through the LSTM Cell</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # The first call to LSTM Cell will create the initial hidden state</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # See that the parameters and states are automatically populated into a field called</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`lstm_cell\` We use \`eachslice\` to get the elements in the sequence without copying,</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # and \`Iterators.peel\` to split out the first element for LSTM initialization.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">_eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Now that we have the hidden state and memory in \`carry\` we will pass the input and</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`carry\` jointly</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry), ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st_lstm)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # After running through the sequence we will pass the output through the classifier</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y, st_classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Finally remember to create the updated state</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> merge</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st, (classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_classifier, lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_lstm))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y), st</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR832/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; lstm_cell, classifier) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">_eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>SpiralClassifierCompact (generic function with 1 method)</span></span></code></pre></div><h2 id="Defining-Accuracy,-Loss-and-Optimiser" tabindex="-1">Defining Accuracy, Loss and Optimiser <a class="header-anchor" href="#Defining-Accuracy,-Loss-and-Optimiser" aria-label="Permalink to &quot;Defining Accuracy, Loss and Optimiser {#Defining-Accuracy,-Loss-and-Optimiser}&quot;">​</a></h2><p>Now let&#39;s define the binarycrossentropy loss. Typically it is recommended to use <code>logitbinarycrossentropy</code> since it is more numerically stable, but for the sake of simplicity we will use <code>binarycrossentropy</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> lossfn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> BinaryCrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> compute_loss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, (x, y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss, st_, (; y_pred</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ŷ)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.5f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_true)</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model_type)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (train_loader, val_loader) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, model, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); transform_variables</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dev)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), lossfn, (x, y), train_state)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, st_)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Validation: Loss %4.5f Accuracy %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56326</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51335</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48202</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44267</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.41710</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40810</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.39004</span></span>
<span class="line"><span>Validation: Loss 0.36033 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37293 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36294</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34955</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33721</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.32102</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30329</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28178</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27479</span></span>
<span class="line"><span>Validation: Loss 0.25279 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.26042 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25806</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24378</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23044</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22181</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20894</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19804</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18811</span></span>
<span class="line"><span>Validation: Loss 0.17680 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18043 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18037</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17044</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16142</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15317</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14719</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13928</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13350</span></span>
<span class="line"><span>Validation: Loss 0.12574 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12800 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12767</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12283</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11456</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10808</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10695</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10058</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09560</span></span>
<span class="line"><span>Validation: Loss 0.09070 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09288 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09351</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08873</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08277</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07947</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07618</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07413</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07344</span></span>
<span class="line"><span>Validation: Loss 0.06623 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06861 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06790</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06512</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06269</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05892</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05799</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05436</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05043</span></span>
<span class="line"><span>Validation: Loss 0.04900 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05134 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04979</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04769</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04729</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04581</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04265</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04116</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03888</span></span>
<span class="line"><span>Validation: Loss 0.03658 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03883 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03673</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03623</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03497</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03522</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03182</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03242</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03089</span></span>
<span class="line"><span>Validation: Loss 0.02776 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02987 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03056</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02805</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02688</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02662</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02362</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02467</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02415</span></span>
<span class="line"><span>Validation: Loss 0.02174 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02360 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02304</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02243</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02216</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02096</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02025</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01891</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02005</span></span>
<span class="line"><span>Validation: Loss 0.01765 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01925 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01957</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01778</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01746</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01691</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01677</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01704</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01555</span></span>
<span class="line"><span>Validation: Loss 0.01483 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01621 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01474</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01490</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01477</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01532</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01451</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01502</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01459</span></span>
<span class="line"><span>Validation: Loss 0.01280 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01402 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01318</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01355</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01297</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01288</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01266</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01244</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01343</span></span>
<span class="line"><span>Validation: Loss 0.01129 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01237 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01279</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01079</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01148</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01124</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01143</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01118</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01204</span></span>
<span class="line"><span>Validation: Loss 0.01010 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01108 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01041</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01075</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01045</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01052</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00966</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01041</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00983</span></span>
<span class="line"><span>Validation: Loss 0.00914 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01003 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00981</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00948</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00950</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00936</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00903</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00919</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00943</span></span>
<span class="line"><span>Validation: Loss 0.00834 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00917 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00896</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00865</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00894</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00857</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00792</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00855</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00870</span></span>
<span class="line"><span>Validation: Loss 0.00766 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00842 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00851</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00769</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00839</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00747</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00802</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00755</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00756</span></span>
<span class="line"><span>Validation: Loss 0.00708 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00779 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00791</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00711</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00734</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00766</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00699</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00698</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00742</span></span>
<span class="line"><span>Validation: Loss 0.00657 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00723 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00707</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00641</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00722</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00696</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00678</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00652</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00678</span></span>
<span class="line"><span>Validation: Loss 0.00612 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00675 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00651</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00654</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00616</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00643</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00610</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00663</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00579</span></span>
<span class="line"><span>Validation: Loss 0.00572 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00631 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00657</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00593</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00593</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00593</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00579</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00590</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00507</span></span>
<span class="line"><span>Validation: Loss 0.00537 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00593 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00621</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00577</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00595</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00528</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00503</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00554</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00525</span></span>
<span class="line"><span>Validation: Loss 0.00505 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00558 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00561</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00490</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00562</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00520</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00544</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00504</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00495</span></span>
<span class="line"><span>Validation: Loss 0.00477 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00527 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56395</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51822</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47832</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44016</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42425</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.41294</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.37693</span></span>
<span class="line"><span>Validation: Loss 0.36756 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37174 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37191</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34320</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.32453</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31885</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30477</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29881</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28585</span></span>
<span class="line"><span>Validation: Loss 0.25807 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.26089 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25583</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24675</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23336</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21780</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21156</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20656</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19006</span></span>
<span class="line"><span>Validation: Loss 0.18040 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18184 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18211</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17339</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16163</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15604</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15063</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14141</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13679</span></span>
<span class="line"><span>Validation: Loss 0.12886 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12972 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12982</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12303</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11799</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11265</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10654</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10474</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09694</span></span>
<span class="line"><span>Validation: Loss 0.09370 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09444 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09486</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08911</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08581</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08314</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07836</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07732</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07022</span></span>
<span class="line"><span>Validation: Loss 0.06907 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06982 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06929</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06553</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06491</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06112</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05793</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05764</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05602</span></span>
<span class="line"><span>Validation: Loss 0.05152 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05226 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05293</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04812</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04736</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04501</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04564</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04328</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04141</span></span>
<span class="line"><span>Validation: Loss 0.03877 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03946 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03903</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03643</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03604</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03444</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03458</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03299</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03250</span></span>
<span class="line"><span>Validation: Loss 0.02961 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03024 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03146</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02742</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02842</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02566</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02649</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02482</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02733</span></span>
<span class="line"><span>Validation: Loss 0.02324 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02380 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02437</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02280</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02215</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02101</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02071</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02030</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01929</span></span>
<span class="line"><span>Validation: Loss 0.01887 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01936 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01898</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01878</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01985</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01746</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01633</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01699</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01460</span></span>
<span class="line"><span>Validation: Loss 0.01585 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01627 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01614</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01636</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01535</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01535</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01435</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01434</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01294</span></span>
<span class="line"><span>Validation: Loss 0.01369 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01407 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01349</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01365</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01320</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01389</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01273</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01284</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01209</span></span>
<span class="line"><span>Validation: Loss 0.01207 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01242 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01174</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01158</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01201</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01193</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01164</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01160</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01204</span></span>
<span class="line"><span>Validation: Loss 0.01081 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01113 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01060</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01064</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01063</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01044</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01067</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01039</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01102</span></span>
<span class="line"><span>Validation: Loss 0.00979 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01008 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01021</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00979</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00965</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00977</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00926</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00917</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00892</span></span>
<span class="line"><span>Validation: Loss 0.00893 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00920 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00956</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00901</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00918</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00868</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00824</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00833</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00792</span></span>
<span class="line"><span>Validation: Loss 0.00820 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00845 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00852</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00851</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00803</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00771</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00818</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00655</span></span>
<span class="line"><span>Validation: Loss 0.00758 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00781 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00774</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00761</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00685</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00751</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00734</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00788</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00770</span></span>
<span class="line"><span>Validation: Loss 0.00704 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00726 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00628</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00708</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00709</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00697</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00714</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00714</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00758</span></span>
<span class="line"><span>Validation: Loss 0.00657 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00677 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00643</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00677</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00665</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00615</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00640</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00685</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00591</span></span>
<span class="line"><span>Validation: Loss 0.00614 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00633 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00611</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00603</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00638</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00661</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00558</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00585</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00628</span></span>
<span class="line"><span>Validation: Loss 0.00576 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00595 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00615</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00595</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00579</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00584</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00529</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00566</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00465</span></span>
<span class="line"><span>Validation: Loss 0.00542 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00559 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00573</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00550</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00543</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00520</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00551</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00519</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00474</span></span>
<span class="line"><span>Validation: Loss 0.00512 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00528 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
<span class="line"><span> :ps_trained</span></span>
<span class="line"><span> :st_trained</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InteractiveUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxDeviceUtils)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDA) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxDeviceUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxCUDADevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPU) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxDeviceUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxAMDGPUDevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        AMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.10.4</span></span>
<span class="line"><span>Commit 48d4fd48430 (2024-06-04 10:41 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LIBM: libopenlibm</span></span>
<span class="line"><span>  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)</span></span>
<span class="line"><span>Threads: 4 default, 0 interactive, 2 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 4</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA runtime 12.5, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.5</span></span>
<span class="line"><span>NVIDIA driver 555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.5.3</span></span>
<span class="line"><span>- CURAND: 10.3.6</span></span>
<span class="line"><span>- CUFFT: 11.2.3</span></span>
<span class="line"><span>- CUSOLVER: 11.6.3</span></span>
<span class="line"><span>- CUSPARSE: 12.5.1</span></span>
<span class="line"><span>- CUPTI: 2024.2.1 (API 23.0.0)</span></span>
<span class="line"><span>- NVML: 12.0.0+555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.4.3</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.9.2+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.14.1+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.4</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2 devices:</span></span>
<span class="line"><span>  0: Quadro RTX 5000 (sm_75, 15.234 GiB / 16.000 GiB available)</span></span>
<span class="line"><span>  1: Quadro RTX 5000 (sm_75, 15.556 GiB / 16.000 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45),h=[l];function e(t,c,k,o,E,r){return n(),a("div",null,h)}const g=s(p,[["render",e]]);export{y as __pageData,g as default};
