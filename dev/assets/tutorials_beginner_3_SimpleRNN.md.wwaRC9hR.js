import{_ as s,c as a,o as n,a4 as i}from"./chunks/framework.CA57S8Fv.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, AMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/dev/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/dev/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/dev/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/dev/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56372</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51471</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47507</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.45431</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42477</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40598</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.37888</span></span>
<span class="line"><span>Validation: Loss 0.35807 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36444 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37210</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35249</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33364</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31471</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30495</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28392</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27364</span></span>
<span class="line"><span>Validation: Loss 0.25153 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25506 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.26538</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24280</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23034</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21683</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20894</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19889</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19268</span></span>
<span class="line"><span>Validation: Loss 0.17662 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17800 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17947</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17165</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16287</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15452</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14634</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13942</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13287</span></span>
<span class="line"><span>Validation: Loss 0.12609 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12674 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12732</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12161</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11625</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11028</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10646</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10079</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09517</span></span>
<span class="line"><span>Validation: Loss 0.09073 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09156 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09180</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08865</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08376</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08205</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07662</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07444</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07119</span></span>
<span class="line"><span>Validation: Loss 0.06599 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06709 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06834</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06512</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06255</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05929</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05834</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05498</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05163</span></span>
<span class="line"><span>Validation: Loss 0.04867 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04984 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05144</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04971</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04752</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04408</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04185</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04076</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04207</span></span>
<span class="line"><span>Validation: Loss 0.03627 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03740 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03899</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03617</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03637</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03342</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03296</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03186</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02775</span></span>
<span class="line"><span>Validation: Loss 0.02745 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02851 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03005</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02861</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02628</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02712</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02510</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02415</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02526</span></span>
<span class="line"><span>Validation: Loss 0.02142 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02238 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02262</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02220</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02146</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02100</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01986</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02115</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02153</span></span>
<span class="line"><span>Validation: Loss 0.01736 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01820 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01890</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01922</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01811</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01749</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01647</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01616</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01607</span></span>
<span class="line"><span>Validation: Loss 0.01456 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01528 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01446</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01607</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01568</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01446</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01406</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01510</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01519</span></span>
<span class="line"><span>Validation: Loss 0.01256 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01320 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01322</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01317</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01322</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01236</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01327</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01322</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01248</span></span>
<span class="line"><span>Validation: Loss 0.01107 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01163 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01249</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01085</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01051</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01197</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01142</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01230</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01145</span></span>
<span class="line"><span>Validation: Loss 0.00990 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01042 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01069</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01044</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01116</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01064</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01033</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00941</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01005</span></span>
<span class="line"><span>Validation: Loss 0.00895 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00942 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00976</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00942</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00935</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00998</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00967</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00898</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00818</span></span>
<span class="line"><span>Validation: Loss 0.00817 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00860 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00896</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00902</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00867</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00855</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00883</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00826</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00773</span></span>
<span class="line"><span>Validation: Loss 0.00750 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00791 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00773</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00857</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00746</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00797</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00797</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00836</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00759</span></span>
<span class="line"><span>Validation: Loss 0.00693 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00731 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00814</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00716</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00746</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00722</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00751</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00707</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00708</span></span>
<span class="line"><span>Validation: Loss 0.00643 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00679 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00731</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00735</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00673</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00651</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00710</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00673</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00545</span></span>
<span class="line"><span>Validation: Loss 0.00599 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00633 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00665</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00649</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00634</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00636</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00657</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00616</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00667</span></span>
<span class="line"><span>Validation: Loss 0.00560 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00592 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00640</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00647</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00546</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00602</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00592</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00601</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00580</span></span>
<span class="line"><span>Validation: Loss 0.00526 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00556 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00576</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00596</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00562</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00562</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00548</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00571</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00526</span></span>
<span class="line"><span>Validation: Loss 0.00495 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00523 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00559</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00572</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00530</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00513</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00490</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00536</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00571</span></span>
<span class="line"><span>Validation: Loss 0.00467 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00494 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56228</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50868</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47706</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.45747</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42428</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40273</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38925</span></span>
<span class="line"><span>Validation: Loss 0.36397 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36650 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36144</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35135</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.32928</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.32457</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29607</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29286</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27677</span></span>
<span class="line"><span>Validation: Loss 0.25539 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25680 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.26251</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24091</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23033</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22036</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20939</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19879</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19586</span></span>
<span class="line"><span>Validation: Loss 0.17895 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17943 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17892</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17006</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16436</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15452</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14767</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14112</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13615</span></span>
<span class="line"><span>Validation: Loss 0.12792 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12809 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12825</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12346</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11696</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11127</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10716</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10130</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09556</span></span>
<span class="line"><span>Validation: Loss 0.09278 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09299 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09350</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08957</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08413</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08198</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07923</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07450</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.06904</span></span>
<span class="line"><span>Validation: Loss 0.06799 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06835 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06776</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06542</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06273</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06146</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05933</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05470</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05607</span></span>
<span class="line"><span>Validation: Loss 0.05043 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05088 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05100</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05113</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04708</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04459</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04297</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04269</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03736</span></span>
<span class="line"><span>Validation: Loss 0.03782 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03827 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03898</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03765</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03454</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03382</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03462</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03172</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03055</span></span>
<span class="line"><span>Validation: Loss 0.02880 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02924 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03002</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02774</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02753</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02579</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02563</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02627</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02523</span></span>
<span class="line"><span>Validation: Loss 0.02259 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02299 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02264</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02025</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02276</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02111</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02181</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02107</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02091</span></span>
<span class="line"><span>Validation: Loss 0.01835 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01870 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02004</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01812</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01785</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01871</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01752</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01540</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01492</span></span>
<span class="line"><span>Validation: Loss 0.01540 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01570 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01603</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01540</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01539</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01538</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01411</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01484</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01319</span></span>
<span class="line"><span>Validation: Loss 0.01330 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01356 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01410</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01318</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01365</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01282</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01341</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01198</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01264</span></span>
<span class="line"><span>Validation: Loss 0.01172 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01196 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01098</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01232</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01209</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01139</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01228</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01134</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01038</span></span>
<span class="line"><span>Validation: Loss 0.01050 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01071 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01162</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01085</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01027</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01045</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01036</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00975</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00966</span></span>
<span class="line"><span>Validation: Loss 0.00950 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00969 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00990</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01013</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00928</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00942</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00966</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00872</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01034</span></span>
<span class="line"><span>Validation: Loss 0.00867 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00885 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00929</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00886</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00843</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00858</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00910</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00849</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00755</span></span>
<span class="line"><span>Validation: Loss 0.00797 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00813 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00803</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00813</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00795</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00791</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00782</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00854</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00768</span></span>
<span class="line"><span>Validation: Loss 0.00736 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00752 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00751</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00793</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00718</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00721</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00757</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00748</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00694</span></span>
<span class="line"><span>Validation: Loss 0.00683 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00698 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00757</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00673</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00682</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00718</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00664</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00661</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00724</span></span>
<span class="line"><span>Validation: Loss 0.00637 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00651 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00674</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00648</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00608</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00671</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00635</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00665</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00584</span></span>
<span class="line"><span>Validation: Loss 0.00596 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00609 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00616</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00641</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00607</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00623</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00579</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00581</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00582</span></span>
<span class="line"><span>Validation: Loss 0.00559 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00571 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00533</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00585</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00582</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00608</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00570</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00542</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00565</span></span>
<span class="line"><span>Validation: Loss 0.00526 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00538 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00524</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00565</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00546</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00543</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00530</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00533</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00455</span></span>
<span class="line"><span>Validation: Loss 0.00496 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00507 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>- CUDA_Driver_jll: 0.9.1+1</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.14.1+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.4</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45),h=[l];function e(t,c,k,o,E,r){return n(),a("div",null,h)}const g=s(p,[["render",e]]);export{y as __pageData,g as default};
